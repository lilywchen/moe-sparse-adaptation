"""Training-only, experiment-stratified FFN gradient-conflict profiling.

The profiler asks where different acquisition experiments request opposing parameter updates.
It never consumes validation or test examples.  Gradients are compressed with a deterministic
CountSketch before pairwise cosine statistics are calculated, keeping the all-layer audit small.
"""
from collections import defaultdict
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def environment_index_groups(dataset):
    """Return raw-environment -> local dataset positions without loading image pixels."""
    subset = getattr(dataset, "subset", None)
    exp_col = getattr(dataset, "exp_col", None)
    if subset is not None and exp_col is not None:
        metadata = getattr(subset, "metadata_array", None)
        if metadata is not None and len(metadata) == len(dataset):
            raw = metadata[:, int(exp_col)]
        elif hasattr(subset, "indices") and hasattr(subset, "dataset"):
            raw = subset.dataset.metadata_array[subset.indices, int(exp_col)]
        else:
            raise TypeError("dataset subset does not expose aligned experiment metadata")
    elif hasattr(dataset, "environment_ids"):
        raw = dataset.environment_ids
    else:
        raise TypeError(
            "dataset must expose (subset, exp_col) or environment_ids for pixel-free grouping")

    groups = defaultdict(list)
    for local_index, environment in enumerate(raw):
        groups[int(environment)].append(local_index)
    if len(groups) < 2:
        raise ValueError("gradient conflict requires at least two training environments")
    return dict(sorted(groups.items()))


def balanced_environment_draws(groups, samples_per_environment=8, rounds=3, seed=0,
                               max_environments=None):
    """Deterministically draw the same number of training examples from every environment."""
    if samples_per_environment < 1 or rounds < 1:
        raise ValueError("samples_per_environment and rounds must be positive")
    environments = sorted(groups)
    if max_environments is not None:
        if int(max_environments) < 2:
            raise ValueError("max_environments must be at least two")
        environments = environments[:int(max_environments)]
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    for _ in range(int(rounds)):
        row = {}
        for environment in environments:
            indices = torch.as_tensor(groups[environment], dtype=torch.long)
            if len(indices) >= samples_per_environment:
                chosen = indices[torch.randperm(len(indices), generator=generator)[
                    :samples_per_environment]]
            else:
                chosen = indices[torch.randint(
                    len(indices), (samples_per_environment,), generator=generator)]
            row[environment] = chosen.tolist()
        draws.append(row)
    return draws


def count_sketch(tensors, sketch_size=4096, seed=0):
    """Compress a list of tensors into one deterministic signed-hash sketch."""
    if sketch_size < 2:
        raise ValueError("sketch_size must be at least two")
    tensors = list(tensors)
    device = next((tensor.device for tensor in tensors if tensor is not None), torch.device("cpu"))
    dtype = next((tensor.dtype for tensor in tensors if tensor is not None), torch.float32)
    out = torch.zeros(int(sketch_size), device=device, dtype=dtype)
    offset = 0
    for tensor in tensors:
        if tensor is None:
            continue
        flat = tensor.detach().reshape(-1)
        keys = torch.arange(flat.numel(), device=device, dtype=torch.int64) + offset
        mixed = keys * 1103515245 + 12345 + int(seed) * 2654435761
        buckets = torch.remainder(mixed, int(sketch_size))
        signs = torch.where(torch.bitwise_and(mixed >> 16, 1) == 0, 1.0, -1.0)
        out.index_add_(0, buckets, flat * signs.to(dtype=flat.dtype))
        offset += flat.numel()
    return out


def pairwise_cosine_metrics(sketches):
    """Summarize pairwise environment-gradient alignment for an [E, D] tensor."""
    if sketches.ndim != 2 or sketches.shape[0] < 2:
        raise ValueError("sketches must have shape [at least 2 environments, sketch width]")
    normalized = F.normalize(sketches.float(), dim=1, eps=1e-12)
    cosine = normalized @ normalized.T
    values = cosine[torch.triu_indices(len(cosine), len(cosine), offset=1).unbind()]
    return {
        "mean_cosine": float(values.mean()),
        "min_cosine": float(values.min()),
        "max_cosine": float(values.max()),
        "conflict_rate": float((values < 0).float().mean()),
        "n_pairs": int(values.numel()),
    }


def _batch(dataset, indices, device):
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices),
                                 shuffle=False, num_workers=0)))
    return batch[0].to(device), batch[1].to(device)


def profile_gradient_conflict(model, dataset, device, samples_per_environment=8, rounds=3,
                              sketch_size=4096, seed=0, max_environments=None):
    """Profile every transformer FFN using balanced batches from training environments only."""
    groups = environment_index_groups(dataset)
    draws = balanced_environment_draws(
        groups, samples_per_environment, rounds, seed, max_environments)
    blocks = list(model.blocks)
    layer_params = [list(block.mlp.parameters()) for block in blocks]
    flat_params = [parameter for params in layer_params for parameter in params]
    spans, cursor = [], 0
    for params in layer_params:
        spans.append((cursor, cursor + len(params)))
        cursor += len(params)

    was_training = model.training
    model.eval()
    per_layer_rounds = [[] for _ in blocks]
    per_layer_norms = [[] for _ in blocks]
    try:
        for round_index, draw in enumerate(draws):
            sketches = [[] for _ in blocks]
            norms = [[] for _ in blocks]
            for environment, indices in draw.items():
                x, y = _batch(dataset, indices, device)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                gradients = torch.autograd.grad(loss, flat_params, allow_unused=True)
                for layer_index, (start, stop) in enumerate(spans):
                    layer_gradients = gradients[start:stop]
                    sketches[layer_index].append(count_sketch(
                        layer_gradients, sketch_size, seed + 1009 * layer_index))
                    sq_norm = sum(float(g.detach().float().square().sum())
                                  for g in layer_gradients if g is not None)
                    norms[layer_index].append(sq_norm ** 0.5)
            for layer_index in range(len(blocks)):
                metrics = pairwise_cosine_metrics(torch.stack(sketches[layer_index]))
                metrics["round"] = round_index
                per_layer_rounds[layer_index].append(metrics)
                per_layer_norms[layer_index].append(mean(norms[layer_index]))
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)

    layers = []
    for block_index, round_metrics in enumerate(per_layer_rounds):
        aggregate = {}
        for key in ("mean_cosine", "min_cosine", "max_cosine", "conflict_rate"):
            values = [row[key] for row in round_metrics]
            aggregate[key] = mean(values)
            aggregate[f"{key}_sd"] = pstdev(values) if len(values) > 1 else 0.0
        layers.append({
            "block_index": block_index,
            "mean_gradient_norm": mean(per_layer_norms[block_index]),
            "rounds": round_metrics,
            **aggregate,
        })

    conflict_ranking = sorted(
        range(len(layers)),
        key=lambda i: (-layers[i]["conflict_rate"], layers[i]["mean_cosine"]),
    )
    aligned_ranking = sorted(
        range(len(layers)),
        key=lambda i: (layers[i]["conflict_rate"], -layers[i]["mean_cosine"]),
    )
    return {
        "n_environments": len(draws[0]),
        "environment_ids": list(draws[0]),
        "samples_per_environment": int(samples_per_environment),
        "rounds": int(rounds),
        "sketch_size": int(sketch_size),
        "layers": layers,
        "conflict_ranking": conflict_ranking,
        "most_conflicted_block": conflict_ranking[0],
        "least_conflicted_block": aligned_ranking[0],
    }
