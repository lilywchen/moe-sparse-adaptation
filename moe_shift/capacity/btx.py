"""Branch-Train-MiX: differentiate experts by CONSTRUCTION rather than by hoping SGD does it.

Every completed MoE arm still starts its experts from one shared initialisation
(``copy.deepcopy(mlp)`` with ``sym_break_moe=0.0``) and relies on gradient descent to separate
them.  The measurements say that does not happen: ``route_reliance <= 0.0065`` against a ``0.01``
gate, and learned routing matched frozen routing.  Identical experts used uniformly satisfy the
canonical balance loss perfectly, so nothing in the objective ever pushed them apart.

BTX (Sukhbaatar et al., 2024) removes the assumption.  It:

1. **Branches** -- partitions the 33 training experiments into ``n_clusters`` groups;
2. **Trains** -- fine-tunes one independent specialist per cluster, on that cluster's data only;
3. **Mixes** -- loads the specialists' FFN weights into an expert bank and trains only the router.

Experts are then guaranteed distinct because they were fitted to disjoint data.  For 33
environments with a computable partition this is an unusually good fit, and it converts the
gradient-conflict profile from a placement heuristic into an expert-assignment prior.

Clustering sources
------------------
``feature_mean``
    Per-environment mean backbone embedding, k-means clustered.  Cheap (one pass over a
    subsample, no gradients) and reflects the representational geometry the router will see.
``gradient_conflict``
    Consumes an existing ``analysis/gradient_conflict_profile_*.json``-style pairwise cosine
    matrix, so the campaign's own conflict measurement chooses which environments share an expert.
``file``
    An explicit ``{"clusters": {"0": [env, ...], ...}}`` mapping, for a frozen hand partition.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .ffn import SharedResidualMoEFFN
from .frontier import LowRankResidualMoEFFN

MANIFEST_NAME = "btx_manifest.json"
CLUSTER_SOURCES = ("feature_mean", "gradient_conflict", "file")


# ------------------------------------------------------------------------------- clustering
def _kmeanspp_centres(points: torch.Tensor, k: int, generator: torch.Generator):
    """k-means++ seeding: each new centre is the point farthest from those already chosen.

    Uniform random seeding is not good enough here.  With 33 environments and k=4 it regularly
    draws two initial centres from the same true group and converges to a local optimum that
    splits one cluster while merging two others.  Since the partition decides which environments
    share an expert, a bad local optimum silently changes the experiment rather than degrading a
    metric.  The farthest-point rule is deterministic after the first draw, which also keeps the
    partition reproducible from the seed.
    """
    n = points.shape[0]
    first = int(torch.randint(n, (1,), generator=generator).item())
    chosen = [first]
    while len(chosen) < k:
        distance = torch.cdist(points, points[chosen]).min(dim=1).values
        distance[torch.tensor(chosen, dtype=torch.long)] = -1.0
        chosen.append(int(distance.argmax().item()))
    return points[chosen].clone()


def _kmeans(points: torch.Tensor, k: int, iters: int = 100, seed: int = 0):
    """Minimal deterministic k-means with k-means++ seeding.

    Deterministic seeding matters: the cluster identity becomes part of the run's scientific
    identity (which environments share an expert), so it must be reproducible from the seed alone.
    """
    n = points.shape[0]
    k = max(1, min(int(k), n))
    generator = torch.Generator().manual_seed(int(seed))
    centres = _kmeanspp_centres(points, k, generator)
    assignment = torch.full((n,), -1, dtype=torch.long)
    for _ in range(int(iters)):
        distance = torch.cdist(points, centres)
        new_assignment = distance.argmin(dim=1)
        if torch.equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for index in range(k):
            member = points[assignment == index]
            if len(member):
                centres[index] = member.mean(dim=0)
    return assignment


def cluster_environments(
    environment_vectors: Dict[int, Sequence[float]],
    n_clusters: int = 4,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """Cluster environments by a per-environment descriptor vector.

    ``environment_vectors`` maps raw environment id -> descriptor.  Returns cluster index ->
    sorted list of raw environment ids.  Empty clusters are dropped and the remaining clusters are
    re-indexed contiguously, so ``n_experts`` always equals ``len(result)``.
    """
    if len(environment_vectors) < 2:
        raise ValueError("BTX clustering requires at least two environments")
    ids = sorted(environment_vectors)
    points = torch.stack([torch.as_tensor(environment_vectors[i], dtype=torch.float32)
                          for i in ids])
    points = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    assignment = _kmeans(points, n_clusters, seed=seed)
    grouped: Dict[int, List[int]] = {}
    for environment, cluster in zip(ids, assignment.tolist()):
        grouped.setdefault(int(cluster), []).append(int(environment))
    return {new: sorted(grouped[old]) for new, old in enumerate(sorted(grouped))}


def cluster_from_conflict_matrix(
    matrix: Dict[str, Dict[str, float]],
    n_clusters: int = 4,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """Cluster environments from a pairwise gradient-cosine matrix.

    Environments that request similar parameter updates belong on the same expert; environments
    that conflict belong on different ones.  Each environment's row of the cosine matrix is used
    as its descriptor, so the clustering is driven by the campaign's own conflict measurement.
    """
    ids = sorted(int(key) for key in matrix)
    vectors = {}
    for environment in ids:
        row = matrix[str(environment)]
        vectors[environment] = [float(row.get(str(other), 0.0)) for other in ids]
    return cluster_environments(vectors, n_clusters=n_clusters, seed=seed)


def load_clusters(path) -> Dict[int, List[int]]:
    """Read a frozen ``{"clusters": {"0": [env, ...]}}`` mapping."""
    payload = json.loads(Path(path).read_text())
    clusters = payload.get("clusters", payload)
    parsed = {int(key): sorted(int(v) for v in value) for key, value in clusters.items()}
    if not parsed:
        raise ValueError(f"no clusters found in {path}")
    return {new: parsed[old] for new, old in enumerate(sorted(parsed))}


# ------------------------------------------------------------------------------- mixing
def _state_for_block(state: Dict[str, torch.Tensor], block_index: int) -> Dict[str, torch.Tensor]:
    """Extract one block's ``mlp.*`` tensors from a full CCASModel state dict.

    Specialists are trained with ``variant=original``, so their FFN lives at the backbone's own
    ``...blocks.<i>.mlp.*`` path.  Cell-DINO checkpoints store blocks inside four ``BlockChunk``
    containers, so the block index can appear at more than one depth; both layouts are matched by
    looking for the ``blocks.<i>.mlp.`` infix rather than assuming a fixed prefix.
    """
    infix = f"blocks.{int(block_index)}.mlp."
    extracted = {}
    for key, value in state.items():
        position = key.find(infix)
        if position >= 0:
            extracted[key[position + len(infix):]] = value
    if not extracted:
        raise KeyError(f"no '{infix}*' tensors found in specialist state dict")
    return extracted


def load_specialist_ffn_states(
    manifest_path,
    block_index: int,
) -> List[Dict[str, torch.Tensor]]:
    """Load each specialist's FFN weights for one block, ordered by cluster index."""
    manifest = json.loads(Path(manifest_path).read_text())
    entries = sorted(manifest["specialists"], key=lambda item: int(item["cluster"]))
    states = []
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"specialist checkpoint missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model", payload)
        states.append(_state_for_block(state, block_index))
    if len(states) < 2:
        raise ValueError("BTX mixing requires at least two specialists")
    return states


def mix_specialists_into_block(block: nn.Module, states: Sequence[Dict[str, torch.Tensor]],
                              freeze_experts: bool = True) -> Dict[str, object]:
    """Overwrite a routed block's expert bank with the specialists' FFN weights.

    Only ``SharedResidualMoEFFN`` is supported as the mixing target: its shared path stays the
    pretrained FFN (so the model is still anchored to Cell-DINO) while each routed expert becomes
    a *differently trained* FFN instead of a copy of the same one.  ``freeze_experts`` holds the
    specialists fixed for the router-only phase, which is the step that actually tests whether a
    router can exploit genuinely distinct experts.

    Note that this deliberately breaks exact function preservation at initialisation: the routed
    residual is no longer zero, because its whole purpose is to inject the specialists' learned
    deviation.  The property is recorded in the returned report rather than silently assumed.
    """
    if isinstance(block, LowRankResidualMoEFFN):
        raise TypeError(
            "low-rank residual experts have no FFN-shaped parameters to receive specialist "
            "weights; mix into a SharedResidualMoEFFN block")
    if not isinstance(block, SharedResidualMoEFFN):
        raise TypeError(f"BTX mixing expects SharedResidualMoEFFN, got {type(block).__name__}")
    if len(states) != len(block.experts):
        raise ValueError(
            f"{len(states)} specialists cannot fill {len(block.experts)} expert slots")

    loaded = []
    for expert, state in zip(block.experts, states):
        missing, unexpected = expert.load_state_dict(state, strict=False)
        if missing:
            raise KeyError(f"specialist state is missing expert tensors: {sorted(missing)[:4]}")
        loaded.append(sorted(unexpected))
    if freeze_experts:
        for expert in block.experts:
            for parameter in expert.parameters():
                parameter.requires_grad_(False)
    return {
        "n_experts_filled": len(states),
        "experts_frozen": bool(freeze_experts),
        "unexpected_keys_per_expert": loaded,
        "function_preserving_at_init": False,
        "reason_not_function_preserving":
            "routed residual carries the specialists' learned deviation by design",
    }


def mix_specialists(model, manifest_path, freeze_experts: bool = True) -> Dict[str, object]:
    """Fill every converted block's expert bank from the specialist manifest."""
    blocks = list(getattr(model, "_moe_blocks", []) or [])
    if not blocks:
        raise ValueError("model has no routed blocks to mix specialists into")
    report = {"blocks": {}, "manifest": str(manifest_path)}
    for block_index, block in zip(model.capacity.block_indices, blocks):
        states = load_specialist_ffn_states(manifest_path, block_index)
        report["blocks"][str(block_index)] = mix_specialists_into_block(
            block, states, freeze_experts=freeze_experts)
    return report


def write_manifest(path, clusters: Dict[int, List[int]], specialists: List[Dict[str, object]],
                   cluster_source: str, extra: Optional[Dict[str, object]] = None):
    """Persist the cluster partition and specialist checkpoint paths."""
    payload = {
        "schema_version": 1,
        "cluster_source": str(cluster_source),
        "n_clusters": len(clusters),
        "clusters": {str(k): list(v) for k, v in clusters.items()},
        "specialists": specialists,
    }
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload
