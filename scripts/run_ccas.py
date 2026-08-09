#!/usr/bin/env python
"""One CCAS run: fixed-parameter-budget dense-vs-sparse adaptation under acquisition shift.

Writes a flat JSON containing everything needed to reproduce and audit the run: the full config,
git SHA + dirty flag, exact parameter accounting (total / router / active) with the budget delta
against the fixed reference P*, protocol assertions, all metrics, and the environment.

    python scripts/run_ccas.py --config configs/ccas_rxrx1.yaml \
        --override model.variant=moe model.placement=middle model.routing_unit=token \
                   model.geometry=cosine model.pressure=canonical seed=0
"""
import argparse, hashlib, json, math, os, platform, socket, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_shift.audit import leakage as audit_leak
from moe_shift.audit import routing as audit_routing
from moe_shift.capacity.model import build_ccas
from moe_shift.capacity.naming import run_id_from
from moe_shift.capacity.surgery import check_budget
from moe_shift.data import make_loaders, make_val_loader
from moe_shift.train.invariance import SiteAdversary, lambda_schedule
from moe_shift.utils.config import apply_overrides, load_config

RESULTS = Path(os.environ.get("MOE_RESULTS", "./RESULTS/ccas"))
HELDOUT_RESULT_FIELDS = (
    "acc_heldout", "worst_env_heldout", "per_env_heldout",
    "per_env_n_heldout", "degradation_gap_test",
)


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             stderr=subprocess.DEVNULL).decode().strip())
        return sha, dirty
    except Exception:
        return "unknown", False


# --------------------------------------------------------------------------- optimisation
def build_optimizer(model, cfg, adversary=None):
    t = cfg["train"]
    llrd = float(t.get("llrd", 0.0) or 0.0)
    base_lr, wd = float(t["optim"]["lr"]), float(t["optim"]["weight_decay"])
    if not (0.0 < llrd < 1.0):
        params = [p for p in model.parameters() if p.requires_grad]
        if adversary is not None:
            params += list(adversary.parameters())
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)
    blocks = list(model.blocks)
    n = len(blocks)
    groups, seen = [], set()
    for i, blk in enumerate(blocks):                      # deeper blocks get a larger LR
        ps = [p for p in blk.parameters() if p.requires_grad]
        seen.update(id(p) for p in ps)
        if ps:
            groups.append({"params": ps, "lr": base_lr * (llrd ** (n - 1 - i)), "weight_decay": wd})
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups.append({"params": rest, "lr": base_lr, "weight_decay": wd})
    if adversary is not None:
        groups.append({"params": list(adversary.parameters()), "lr": base_lr, "weight_decay": wd})
    return torch.optim.AdamW(groups)


OBJECTIVES = ("erm", "environment_balanced", "group_dro")


def classification_objective(logits, labels, environments, objective="erm",
                             label_smoothing=0.0, group_dro=None):
    """Return the supervised loss while making environment weighting explicit.

    ``environment_balanced`` gives every experiment represented in the minibatch equal weight,
    rather than letting experiments with more images dominate the update.  It uses no validation
    or test information and changes no model capacity.

    ``group_dro`` optimises the WORST experiment rather than the mean (Sagawa et al., 2020).  The
    completed campaigns' most reproducible MoE effect is a mean/tail trade -- replacement depth
    cost ``1.481`` OOD-test points while raising worst-batch accuracy ``2.377`` points -- so this
    objective makes that trade the target instead of a side effect.  It requires the stateful
    :class:`GroupDRO` weight tracker, which the caller owns across steps.

    ``label_smoothing`` matters mechanically here, not just as regularisation: train accuracy
    reaches ``1.0`` by epoch 30, and the straight-through router gradient is scaled by
    ``d(loss)/d(topv)``.  A saturated cross-entropy therefore stops training the router entirely,
    leaving only the balance loss to shape routing.  Smoothing keeps the task gradient alive.
    """
    per_example = F.cross_entropy(
        logits, labels, reduction="none", label_smoothing=float(label_smoothing))
    if objective == "erm":
        return per_example.mean()
    if objective == "group_dro":
        if group_dro is None:
            raise ValueError("train.objective=group_dro requires a GroupDRO tracker")
        return group_dro(per_example, environments)
    if objective != "environment_balanced":
        raise ValueError(f"unknown train.objective: {objective!r}")
    present = torch.unique(environments)
    if torch.any(present < 0):
        raise ValueError("environment_balanced training requires non-negative train environments")
    return torch.stack([per_example[environments == env].mean() for env in present]).mean()


class GroupDRO:
    """Online group-DRO weights over training environments (Sagawa et al., 2020).

    Group weights follow exponentiated gradient ascent on the per-group loss::

        q_g <- q_g * exp(step_size * loss_g)        (then renormalised over all groups)

    Only groups PRESENT in the current minibatch are updated, and the returned loss is their
    ``q``-weighted mean renormalised over the present groups.  With 33 training experiments and
    batch 64, most groups are absent from any given minibatch, so renormalising over the present
    subset is what keeps the objective an unbiased reweighting rather than a silently shrinking
    loss.

    ``q`` is deliberately kept out of the optimiser: it is a dual variable, not a parameter, and
    it is recorded in the run JSON so the realised weighting is auditable after the fact.
    """

    def __init__(self, n_groups: int, step_size: float = 0.01, device=None):
        if int(n_groups) < 2:
            raise ValueError("group DRO requires at least two training environments")
        if float(step_size) <= 0:
            raise ValueError("group DRO step_size must be positive")
        self.n_groups = int(n_groups)
        self.step_size = float(step_size)
        self.q = torch.ones(self.n_groups, dtype=torch.float32, device=device) / self.n_groups

    def __call__(self, per_example: torch.Tensor, environments: torch.Tensor) -> torch.Tensor:
        present = torch.unique(environments)
        if torch.any(present < 0):
            raise ValueError("group DRO requires non-negative train environment ids")
        if torch.any(present >= self.n_groups):
            raise ValueError(
                f"environment id >= n_groups ({self.n_groups}); check cfg['sites']['K']")
        self.q = self.q.to(per_example.device)
        group_losses = torch.stack([per_example[environments == g].mean() for g in present])
        with torch.no_grad():
            index = present.to(self.q.device)
            self.q[index] = self.q[index] * torch.exp(
                self.step_size * group_losses.detach().to(self.q.dtype))
            self.q = self.q / self.q.sum().clamp_min(1e-12)
        weights = self.q[index].to(group_losses.dtype)
        weights = weights / weights.sum().clamp_min(1e-12)
        return (weights * group_losses).sum()

    def state(self):
        """Serialisable snapshot of the realised group weighting."""
        values = self.q.detach().float().cpu().tolist()
        return {
            "n_groups": self.n_groups,
            "step_size": self.step_size,
            "weights": values,
            "max_weight": max(values) if values else None,
            "min_weight": min(values) if values else None,
        }


def cross_experiment_contrastive_loss(features, labels, environments, temperature=0.1):
    """Supervised contrastive loss with positives from distinct source experiments.

    The paired RxRx1 sampler guarantees that same-label positives are also from the same cell
    type.  This loss itself remains generic: it receives no test metadata and simply excludes
    same-experiment pairs from the positive set.
    """
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    z = F.normalize(features.float(), dim=-1)
    logits = (z @ z.t()) / float(temperature)
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    valid_pair = ~diagonal
    positives = (
        labels[:, None].eq(labels[None, :])
        & environments[:, None].ne(environments[None, :])
        & valid_pair
    )
    positive_count = positives.sum(dim=1)
    valid_anchor = positive_count > 0
    if not bool(valid_anchor.any()):
        return features.sum() * 0.0
    logits = logits.masked_fill(~valid_pair, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1)
    return -(positive_log_prob[valid_anchor] / positive_count[valid_anchor]).mean()


def router_gradient_norms(loss, model):
    """Classification-loss gradient norm for each trainable router, without accumulating grads.

    This is a fail-fast audit for hard top-1 routing. A selected-ST run must expose a finite,
    nonzero task gradient before auxiliary losses are added; the historical renormalised top-1
    implementation did not. Frozen-router, oracle-routed and non-MoE models correctly return an
    empty mapping -- ``oracle_moe`` has no learned router at all, and ``soft_moe``'s slot
    projection is reported under the same key because it plays the router's role.

    Logged EVERY epoch, not only at initialisation: train accuracy reaches 1.0 by epoch 30, so
    the interesting quantity is when this norm decays to zero, which a single init-time probe
    cannot show.
    """
    norms = {}
    for block_index, block in zip(model.capacity.block_indices, model.moe_blocks):
        router = getattr(block, "router", None)
        if router is not None:
            params = [p for p in router.parameters() if p.requires_grad]
        else:
            # Soft MoE routes through `phi`; oracle blocks have no routing parameters at all.
            phi = getattr(block, "phi", None)
            params = [phi] if phi is not None and phi.requires_grad else []
        if not params:
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        squared = loss.new_zeros(())
        for grad in grads:
            if grad is not None:
                squared = squared + grad.detach().float().square().sum()
        norms[str(block_index)] = float(squared.sqrt())
    return norms


def _routing_parameters(block):
    """The parameters that play the router's role for this block, or [] if it has none.

    ``oracle_moe`` routes on ground truth and has no routing parameters; ``soft_moe`` routes
    through its slot projection ``phi``.  Returning a list keeps every caller uniform instead of
    each one re-deriving the special cases.
    """
    router = getattr(block, "router", None)
    if router is not None:
        return list(router.parameters())
    phi = getattr(block, "phi", None)
    return [phi] if phi is not None else []


def snapshot_routers(model):
    """Small CPU snapshots used to prove that trainable routers actually moved."""
    return {
        str(block_index): [p.detach().float().cpu().clone()
                           for p in _routing_parameters(block)]
        for block_index, block in zip(model.capacity.block_indices, model.moe_blocks)
    }


def router_parameter_deltas(model, initial):
    """Absolute and relative L2 displacement of every router from initialisation."""
    deltas = {}
    for block_index, block in zip(model.capacity.block_indices, model.moe_blocks):
        key = str(block_index)
        before = initial.get(key)
        if not before:
            continue
        delta_sq = base_sq = 0.0
        for current, start in zip(_routing_parameters(block), before):
            current = current.detach().float().cpu()
            delta_sq += float((current - start).square().sum())
            base_sq += float(start.square().sum())
        absolute = math.sqrt(delta_sq)
        deltas[key] = {
            "l2": absolute,
            "relative_l2": absolute / max(math.sqrt(base_sq), 1e-12),
        }
    return deltas


def milestone_epochs(cfg):
    """Validate and return sorted one-indexed milestone/checkpoint epochs."""
    total = int(cfg["train"]["epochs"])
    milestones = sorted({int(e) for e in cfg["train"].get("milestone_epochs", [])})
    checkpoints = sorted({int(e) for e in cfg["train"].get("save_checkpoint_epochs", [])})
    if any(e < 1 or e > total for e in milestones + checkpoints):
        raise ValueError("milestone/checkpoint epochs must lie in [1, train.epochs]")
    if not set(checkpoints).issubset(milestones):
        raise ValueError("save_checkpoint_epochs must be a subset of milestone_epochs")
    return milestones, checkpoints


def atomic_torch_save(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_withheld_ood_fields(result):
    """Normalize legacy empty OOD-test maps to explicit nulls without touching metrics."""
    normalized = dict(result)
    if normalized.get("test_evaluated") is not False:
        return normalized
    for key in HELDOUT_RESULT_FIELDS:
        value = normalized.get(key)
        if value not in (None, {}):
            raise ValueError(f"cannot normalize non-empty OOD-test field {key}")
        normalized[key] = None
    return normalized


def validate_stage1_artifacts(result, milestone_path=None):
    """Fail closed before publishing any selection-stage artifact externally."""
    if result.get("selection_split") != "ood_val" or result.get("test_evaluated") is not False:
        raise ValueError("publish requires selection_split=ood_val and test_evaluated=false")
    if any(result.get(key) is not None for key in HELDOUT_RESULT_FIELDS):
        raise ValueError("publish requires all OOD-test metrics to remain null")
    required = ("acc_selection", "acc_val", "worst_env_val", "acc_within")
    if any(not math.isfinite(float(result[key])) for key in required):
        raise ValueError("publish requires finite selection/ID metrics")
    milestones = []
    if milestone_path is not None and Path(milestone_path).is_file():
        with open(milestone_path) as handle:
            milestones = [json.loads(line) for line in handle if line.strip()]
        for row in milestones:
            if row.get("run_id") != result.get("run_id"):
                raise ValueError("milestone run identity mismatch")
            if row.get("selection_split") != "ood_val" or row.get("test_evaluated") is not False:
                raise ValueError("milestone violates OOD-test blindness")
            for key in ("acc_train", "acc_within", "acc_selection", "worst_env_val"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"milestone has non-finite {key}")
    return milestones


def validate_publishable_artifacts(result, milestone_path=None):
    """Validate either a test-blind selection run or an explicitly declared stage-3 run."""
    if result.get("selection_split") != "ood_val":
        raise ValueError("publish requires selection on ood_val")
    required = ("acc_selection", "acc_val", "worst_env_val", "acc_within")
    if any(not math.isfinite(float(result[key])) for key in required):
        raise ValueError("publish requires finite selection/ID metrics")

    if result.get("test_evaluated") is True:
        if int(result.get("stage", 0)) < 3:
            raise ValueError("test-evaluated publication requires stage >= 3")
        for key in ("acc_heldout", "worst_env_heldout"):
            if not math.isfinite(float(result[key])):
                raise ValueError(f"publish requires finite {key}")
        if not result.get("per_env_heldout") or not result.get("per_env_n_heldout"):
            raise ValueError("publish requires per-environment OOD-test results")
    elif result.get("test_evaluated") is False:
        if any(result.get(key) is not None for key in HELDOUT_RESULT_FIELDS):
            raise ValueError("test-blind publication requires null OOD-test metrics")
    else:
        raise ValueError("test_evaluated must be explicitly true or false")

    milestones = []
    if milestone_path is not None and Path(milestone_path).is_file():
        with open(milestone_path) as handle:
            milestones = [json.loads(line) for line in handle if line.strip()]
        for row in milestones:
            if row.get("run_id") != result.get("run_id"):
                raise ValueError("milestone run identity mismatch")
            if row.get("selection_split") != "ood_val" or row.get("test_evaluated") is not False:
                raise ValueError("milestones must remain OOD-test blind")
            for key in ("acc_train", "acc_within", "acc_selection", "worst_env_val"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"milestone has non-finite {key}")
    return milestones


def publish_hf_run(result, artifact_paths, out_dir):
    """Upload one validated run folder plus a checksum manifest to the configured HF dataset."""
    token, repo_id = os.environ.get("HF_TOKEN"), os.environ.get("CCAS_HF_REPO")
    if not token or not repo_id:
        return None
    from huggingface_hub import HfApi

    prefix = os.environ.get("CCAS_HF_PREFIX", "results").strip("/")
    run_id = result["run_id"]
    files = [Path(path) for path in artifact_paths if Path(path).is_file()]
    manifest_path = Path(out_dir) / f"{run_id}.artifact_manifest.json"
    manifest = {
        "run_id": run_id, "selection_split": result["selection_split"],
        "test_evaluated": result["test_evaluated"], "git_sha": result["git_sha"],
        "files": [{"name": path.name, "bytes": path.stat().st_size,
                   "sha256": _sha256_file(path)} for path in files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    files.append(manifest_path)
    api = HfApi(token=token)
    for path in files:
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo=f"{prefix}/{run_id}/{path.name}",
            repo_id=repo_id, repo_type="dataset",
        )
    return {"prefix": f"{prefix}/{run_id}", "manifest": manifest_path.name,
            "n_files": len(files)}


def batch_group_ids(batch, group_source):
    """Oracle group ids for this batch, or None when the arm does not use oracle routing.

    ``cell_type``   -> ``batch[4]``, the RxRx1 cell line.  Defined on every split, including the
                       unseen OOD experiments, so oracle cell-type experts really do apply there.
    ``environment`` -> ``batch[2]``, the TRAIN-remapped site index.  This is ``-1`` on every OOD
                       row, so an unseen experiment matches no expert and falls through to the
                       shared path automatically -- which is exactly the ceiling being measured.
    """
    if not group_source:
        return None
    if group_source == "cell_type":
        if len(batch) < 5:
            raise ValueError(
                "oracle cell-type routing needs a 5-element loader batch (x, y, site, env, cell); "
                "this dataset does not expose cell_type")
        return batch[4]
    if group_source == "environment":
        return batch[2]
    raise ValueError(f"unknown group_source: {group_source!r}")


@torch.no_grad()
def evaluate(model, loader, device, group_source=None):
    """-> (acc, worst_env_acc, per_env_acc, per_env_n), bucketed by RAW environment id.

    `per_env_n` is required, not cosmetic: the plan's uncertainty is a CLUSTER bootstrap over
    experiments / hospitals, which cannot be reconstructed from per-environment accuracies alone.

    ``group_source`` forwards oracle group ids to the model.  Evaluation must do this, not just
    training: an oracle block raises rather than silently guessing when its group ids are stale.
    """
    model.eval()
    ok = tot = 0
    per_env_ok, per_env_tot = {}, {}
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        # Bucket by the RAW environment id (batch[3]) when the loader provides one. batch[2] is
        # the TRAIN-remapped site index, which is -1 for every row of an OOD split -- keying on it
        # collapsed per_env to a single '-1' bucket and made worst_env_* equal to overall accuracy.
        # The 3-tuple fallback is for the injected-nuisance loaders, where site IS the environment.
        s = batch[3] if len(batch) > 3 else batch[2]
        group = batch_group_ids(batch, group_source)
        if group is not None:
            model.set_group(group.to(device))
        pred = model(x).argmax(1)
        c = (pred == y)
        ok += c.sum().item(); tot += y.numel()
        for e in torch.unique(s):
            m = (s == e)
            k = int(e)
            per_env_ok[k] = per_env_ok.get(k, 0) + c[m.to(c.device)].sum().item()
            per_env_tot[k] = per_env_tot.get(k, 0) + int(m.sum())
    acc = ok / max(tot, 1)
    per_env = {k: per_env_ok[k] / max(per_env_tot[k], 1) for k in per_env_ok}
    worst = min(per_env.values()) if per_env else float("nan")
    return acc, worst, per_env, dict(per_env_tot)


@torch.no_grad()
def counterfactual_reroute(model, loader, device, seed=0, group_source=None):
    """Joint route reliance with every converted router randomized and expert weights fixed.

    The router is temporarily replaced by i.i.d. Gaussian logits, which give a balanced random
    top-1 assignment while leaving every expert's weights untouched. The substitution is always
    undone (finally), and the RNG is seeded so the number is reproducible across reruns.

    Blocks with no learned router are skipped rather than crashing the audit.  For ``soft_moe``
    the slot projection ``phi`` plays the router's role, so randomising its logits is the matching
    counterfactual; for ``oracle_moe`` there is nothing to randomise, and the informative
    counterfactual is :func:`shared_only_accuracy` instead.  ``None`` is returned when no block
    had a randomisable router, so a caller can tell "no reliance" apart from "not measurable".
    """
    blocks = [b for b in model.moe_blocks
              if getattr(b, "router", None) is not None or getattr(b, "phi", None) is not None]
    if not blocks:
        return None
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    saved = []

    def randomized_forward(block, n_out):
        def forward(z):
            r = torch.randn(z.shape[0], n_out, generator=gen)
            return r.to(z.device, dtype=z.dtype)
        return forward

    try:
        for block in blocks:
            router = getattr(block, "router", None)
            if router is not None:
                saved.append((block, "router", router.forward))
                router.forward = randomized_forward(block, block.n_experts)
            else:
                # Soft MoE: replace phi with a fresh random projection of identical shape, so the
                # slot assignment is scrambled while every expert's weights stay untouched.
                saved.append((block, "phi", block.phi.detach().clone()))
                block.phi.copy_(torch.randn(
                    block.phi.shape, generator=gen).to(block.phi.device, block.phi.dtype)
                    * block.phi.shape[0] ** -0.5)
        acc = evaluate(model, loader, device, group_source=group_source)[0]
    finally:
        for block, kind, value in saved:
            if kind == "router":
                block.router.forward = value
            else:
                block.phi.copy_(value.to(block.phi.device, block.phi.dtype))
    return acc


@torch.no_grad()
def shared_only_accuracy(model, loader, device, group_source=None):
    """Accuracy with every supported residual/oracle block forced through its shared path alone.

    For oracle routing this is the ceiling's held-out-group readout.  For shared residual MoE it
    removes the learned correction without changing any weights, directly measuring whether the
    residual branch contributes to the selected-split accuracy.  Returns ``None`` when no block
    supports the ablation.
    """
    if not any(hasattr(b, "shared_only") for b in model.moe_blocks):
        return None
    toggled = model.set_shared_only(True)
    try:
        if not toggled:
            return None
        return evaluate(model, loader, device, group_source=group_source)[0]
    finally:
        model.set_shared_only(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    milestones, checkpoint_epochs = milestone_epochs(cfg)
    pressure = cfg["model"].get("pressure", "canonical")
    if pressure not in ("canonical", "route", "output"):
        raise ValueError(f"model.pressure must be canonical|route|output, got {pressure!r}")
    expected_balance = "within_environment" if pressure == "route" else "global"
    cfg["model"]["balance"] = expected_balance
    rid = run_id_from(cfg)
    out_dir = Path(args.results_dir) if args.results_dir else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{rid}.json"
    if out_json.exists():
        print(f"[skip] {rid} already done"); return

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    train_loader, test_within, test_heldout, audit_loader = make_loaders(cfg)
    val_loader = make_val_loader(cfg)

    model = build_ccas(cfg).to(device)
    cap = model.capacity

    # BTX phase 3: replace the expert bank with independently trained specialists. Done BEFORE the
    # router snapshot so router displacement is measured from the mixed model, and recorded in the
    # protocol because it deliberately breaks function preservation at initialisation.
    btx_report = None
    btx_manifest = cfg["model"].get("btx_manifest")
    if btx_manifest:
        from moe_shift.capacity.btx import mix_specialists
        btx_report = mix_specialists(
            model, btx_manifest,
            freeze_experts=bool(cfg["model"].get("btx_freeze_experts", True)))
        print(f"[btx] mixed specialists: {btx_report}", flush=True)

    initial_routers = snapshot_routers(model)
    inv_target = float(cfg["losses"].get("invariance_w", 0.0)) if pressure == "output" else 0.0
    if pressure == "output" and inv_target <= 0:
        raise ValueError("pressure=output requires losses.invariance_w > 0")
    adversary = (SiteAdversary(model.dim, int(cfg["sites"]["K"])).to(device)
                 if pressure == "output" else None)

    # Start tracking before optimization so long SciServer runs expose live progress. Tracking is
    # deliberately non-fatal: the persistent JSON/JSONL files remain the source of truth.
    wandb_run = None
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline":
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "ccas"),
                entity=os.environ.get("WANDB_ENTITY"), name=rid,
                group=os.environ.get("WANDB_GROUP", f"{cfg['dataset']}_{cap.variant}"),
                job_type=os.environ.get("WANDB_JOB_TYPE", "train"),
                tags=[tag for tag in os.environ.get("WANDB_TAGS", "").split(",") if tag],
                config={**cfg, "capacity": cap.as_dict()}, reinit=True,
            )
        except Exception as e:
            print(f"[wandb] live tracking unavailable: {e}", flush=True)

    # ---- protocol assertions (recorded, and fatal if violated) ----
    protocol = {}
    ffn_only = cap.ffn_block_params - cap.router_params
    p_star = cap.total_params + (0 if cap.router_params else 0)
    protocol["variant"] = cap.variant
    protocol["block_index"] = cap.block_index
    protocol["block_indices"] = list(cap.block_indices)
    protocol["n_blocks"] = len(model.blocks)
    protocol["n_blocks_converted"] = cap.n_converted_blocks
    protocol["exactly_one_block_converted"] = cap.n_converted_blocks == 1
    protocol["training_pressure"] = pressure
    protocol["route_balance"] = expected_balance
    protocol["output_adversary"] = adversary is not None
    protocol["classification_objective"] = str(cfg["train"].get("objective", "erm"))
    protocol["milestone_epochs"] = milestones
    protocol["checkpoint_epochs"] = checkpoint_epochs
    if cap.variant in ("moe", "moe_frozen"):
        protocol["experts_are_upcycled_copies"] = True
        protocol["router_trainable"] = (cap.variant == "moe")
        protocol["routing_estimator"] = str(
            cfg["model"].get("routing_estimator", "selected_st"))
    elif cap.variant == "shared_moe":
        protocol["pretrained_shared_expert_always_active"] = True
        protocol["residual_experts_zero_output_initialized"] = True
        protocol["router_trainable"] = True
        protocol["routing_estimator"] = str(
            cfg["model"].get("routing_estimator", "selected_st"))
    elif cap.variant in ("oracle_moe", "condln_moe", "soft_moe", "lowrank_moe"):
        # Every frontier variant keeps the pretrained dense FFN active for all inputs: replacement
        # is the one design the completed campaigns showed to be actively harmful.
        protocol["pretrained_shared_expert_always_active"] = True
        protocol["frontier_variant"] = cap.variant
        if cap.variant == "oracle_moe":
            protocol["router_trainable"] = False
            protocol["routing_is_ground_truth"] = True
            protocol["group_source"] = str(cfg["model"].get("group_source", "cell_type"))
            protocol["expert_dropout"] = float(cfg["model"].get("expert_dropout", 0.5))
            protocol["deployable"] = False          # this arm is a CEILING, not a method
        if cap.variant == "condln_moe":
            protocol["router_trainable"] = True
            protocol["expert_form"] = "layernorm_affine"
            protocol["condln_descriptor"] = str(
                cfg["model"].get("condln_descriptor", "token_stats"))
            protocol["transductive"] = False        # descriptor comes from the input alone
        if cap.variant == "soft_moe":
            protocol["router_trainable"] = True
            protocol["discrete_routing"] = False
            protocol["auxiliary_balance_loss_active"] = False
            protocol["all_experts_active"] = True
            protocol["softmax_scope"] = "within_image"
        if cap.variant == "lowrank_moe":
            protocol["router_trainable"] = True
            protocol["expert_rank"] = int(cfg["model"].get("expert_rank", 16))
            protocol["diversity_w"] = float(cfg["model"].get("diversity_w", 0.0))
            protocol["routing_estimator"] = str(
                cfg["model"].get("routing_estimator", "selected_st"))

    # Oracle arms need ground-truth group ids at train AND eval time; every other arm passes None
    # so the loaders and evaluation path stay on exactly one code path.
    group_source = (str(cfg["model"].get("group_source", "cell_type"))
                    if cap.variant == "oracle_moe" else None)
    protocol["group_source"] = group_source
    if btx_report is not None:
        protocol["btx"] = btx_report
        protocol["experts_are_upcycled_copies"] = False
        protocol["experts_are_independent_specialists"] = True
        protocol["function_preserving_at_init"] = False
    if cfg["train"].get("environment_subset"):
        # A specialist is not comparable with a full-data arm; make that unmistakable in the JSON.
        protocol["environment_subset"] = [int(e) for e in cfg["train"]["environment_subset"]]
        protocol["trained_on_environment_subset"] = True

    opt = build_optimizer(model, cfg, adversary)
    epochs = int(cfg["train"]["epochs"])
    warm = int(cfg["train"].get("warmup_epochs", 0))
    sched = None
    if warm > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=warm),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warm))],
            milestones=[warm])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bw = float(cfg["losses"]["balance_w"]); zw = float(cfg["losses"].get("zloss_w", 0.0))
    log_path = out_dir / f"{rid}.trainlog.jsonl"
    logf = open(log_path, "a")
    milestone_path = out_dir / f"{rid}.milestones.jsonl"
    milestone_f = open(milestone_path, "w") if milestones else None
    objective = str(cfg["train"].get("objective", "erm"))
    consistency_w = float(cfg["losses"].get("cross_experiment_contrastive_w", 0.0))
    consistency_temperature = float(
        cfg["losses"].get("cross_experiment_contrastive_temperature", 0.1))
    paired_batches = bool(cfg["train"].get("cross_experiment_pairs", False))
    if consistency_w < 0:
        raise ValueError("cross-experiment contrastive weight cannot be negative")
    if consistency_w > 0 and not paired_batches:
        raise ValueError(
            "cross-experiment contrastive loss requires train.cross_experiment_pairs=true")
    protocol["cross_experiment_pairs"] = paired_batches
    protocol["cross_experiment_contrastive_w"] = consistency_w

    if objective not in OBJECTIVES:
        raise ValueError(f"train.objective must be one of {OBJECTIVES}, got {objective!r}")
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0) or 0.0)
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("train.label_smoothing must lie in [0, 1)")
    protocol["label_smoothing"] = label_smoothing
    group_dro = None
    if objective == "group_dro":
        group_dro = GroupDRO(
            int(cfg["sites"]["K"]),
            step_size=float(cfg["train"].get("group_dro_step_size", 0.01)),
            device=device)
        protocol["group_dro_step_size"] = group_dro.step_size
        protocol["group_dro_n_groups"] = group_dro.n_groups

    # Dense-to-sparse annealing: start with every expert active so each one is supervised while
    # data is still plentiful, then contract to the target top-k. `anneal_top_k_epochs` is the
    # epoch by which the target is reached; 0 disables annealing entirely.
    anneal_epochs = int(cfg["train"].get("anneal_top_k_epochs", 0) or 0)
    if anneal_epochs < 0:
        raise ValueError("train.anneal_top_k_epochs cannot be negative")
    target_top_k = int(cfg["model"].get("top_k", 1))
    n_experts_cfg = int(cfg["model"]["n_experts"])
    protocol["anneal_top_k_epochs"] = anneal_epochs
    if anneal_epochs:
        applied = model.set_top_k(n_experts_cfg)
        if not applied:
            raise ValueError(
                f"train.anneal_top_k_epochs is set but variant {cap.variant!r} has no settable "
                "top-k; only lowrank_moe supports dense-to-sparse annealing")
        protocol["anneal_top_k_from"] = n_experts_cfg
        protocol["anneal_top_k_to"] = target_top_k

    initial_router_task_grad_norms = None
    router_grad_norm_by_epoch = {}
    group_dro_weight_trace = {}
    for ep in range(epochs):
        model.train()
        if adversary is not None:
            adversary.train()
        if anneal_epochs:
            # Linear contraction in k from n_experts down to the target, reached at anneal_epochs.
            fraction = min(1.0, ep / max(anneal_epochs, 1))
            k_now = int(round(n_experts_cfg + fraction * (target_top_k - n_experts_cfg)))
            model.set_top_k(k_now)
        epoch_router_norms = None
        run_loss = run_aux = run_adv = run_consistency = 0.0; nb = 0
        for batch in train_loader:
            x, y, s = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            model.set_env(s)                       # used ONLY by within-environment balancing
            group = batch_group_ids(batch, group_source)
            if group is not None:
                model.set_group(group.to(device))
            feats = model.forward_features(x)
            logits = model.fc(feats)
            loss = classification_objective(
                logits, y, s, objective, label_smoothing=label_smoothing, group_dro=group_dro)
            consistency = (
                cross_experiment_contrastive_loss(
                    feats, y, s, temperature=consistency_temperature)
                if consistency_w > 0 else loss.new_zeros(())
            )
            if initial_router_task_grad_norms is None and model.moe_blocks:
                probed_norms = router_gradient_norms(loss, model)
                # Shared residual experts have exact-zero outputs on their first minibatch, so
                # their task gradient to the router is correctly zero until one expert update.
                # Probe again on minibatch two; a persistent zero is then visible rather than
                # being mislabeled as the expected function-preserving initialization.
                defer_zero_initialized_probe = (
                    cap.variant == "shared_moe" and nb == 0
                    and not any(value > 0 for value in probed_norms.values())
                )
                if defer_zero_initialized_probe:
                    print("[router-gradient-check] deferred until residual experts leave zero init",
                          flush=True)
                else:
                    initial_router_task_grad_norms = probed_norms
                    print(
                        "[router-gradient-check] classification-only norms "
                        f"{initial_router_task_grad_norms}", flush=True)
            # Per-EPOCH router gradient norm. The single init-time probe above cannot show the
            # failure that matters: train accuracy reaches 1.0 by epoch 30, and the selected-ST
            # estimator's gradient is scaled by d(loss)/d(topv), so a saturated task loss stops
            # training the router while the balance loss keeps shaping it. Recording the norm
            # every epoch makes that decay visible instead of inferred.
            if epoch_router_norms is None and model.moe_blocks:
                epoch_router_norms = router_gradient_norms(loss, model)
            aux = model.aux_loss(bw, zw)
            adv = loss.new_zeros(())
            if adversary is not None:
                lambd = lambda_schedule(ep, epochs, inv_target)
                adv = F.cross_entropy(adversary(feats, lambd), s)
            (loss + consistency_w * consistency + aux + adv).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            run_loss += loss.item(); run_aux += float(aux); run_adv += float(adv)
            run_consistency += float(consistency); nb += 1
        sched.step()
        if epoch_router_norms:
            router_grad_norm_by_epoch[str(ep)] = epoch_router_norms
        if group_dro is not None:
            group_dro_weight_trace[str(ep)] = group_dro.state()["max_weight"]
        rec = {"epoch": ep, "loss": run_loss / max(nb, 1), "aux": run_aux / max(nb, 1),
               "adversary": run_adv / max(nb, 1),
               "cross_experiment_contrastive": run_consistency / max(nb, 1),
               "router_grad_norms": epoch_router_norms,
               "active_top_k": (model.moe_blocks[0].top_k if model.moe_blocks else None),
               "lr": opt.param_groups[-1]["lr"], "t": round(time.time() - t0, 1)}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if wandb_run is not None:
            try:
                # Only scalars go to the tracker: `router_grad_norms` is a per-block mapping and
                # belongs in the JSONL, which is the source of truth anyway.
                wandb_run.log({f"train/{k}": v for k, v in rec.items()
                               if k != "epoch" and isinstance(v, (int, float))},
                              step=ep)
            except Exception as e:
                print(f"[wandb] epoch log skipped: {e}", flush=True)
        print(f"[{rid}] ep{ep} loss {rec['loss']:.4f} aux {rec['aux']:.4f} "
              f"adv {rec['adversary']:.4f} xbc {rec['cross_experiment_contrastive']:.4f}",
              flush=True)

        epoch_number = ep + 1
        if epoch_number in milestones:
            if val_loader is None:
                raise RuntimeError("milestone evaluation requires a validation split; OOD test fallback is forbidden")
            # Milestone train evaluation iterates the augmented loader.  Preserve the main-process
            # RNG so evaluation does not change the subsequent optimization trajectory.
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            np_rng = np.random.get_state()
            try:
                m_train, m_worst_train, _, _ = evaluate(
                    model, train_loader, device, group_source=group_source)
                m_id, m_worst_id, _, _ = evaluate(
                    model, test_within, device, group_source=group_source)
                m_val, m_worst_val, m_per_env, m_per_env_n = evaluate(
                    model, val_loader, device, group_source=group_source)
            finally:
                torch.random.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                np.random.set_state(np_rng)
            milestone = {
                "run_id": rid, "epoch": epoch_number, "seed": cfg["seed"],
                "acc_train": m_train, "worst_env_train": m_worst_train,
                "acc_within": m_id, "worst_env_within": m_worst_id,
                "acc_selection": m_val, "acc_val": m_val, "worst_env_val": m_worst_val,
                "per_env_val": m_per_env, "per_env_n_val": m_per_env_n,
                "selection_split": "ood_val", "test_evaluated": False,
                "classification_objective": objective,
            }
            if epoch_number in checkpoint_epochs:
                ckpt_path = out_dir / f"{rid}.epoch{epoch_number:03d}.pt"
                atomic_torch_save({
                    "run_id": rid, "epoch": epoch_number, "config": cfg,
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "milestone": milestone,
                }, ckpt_path)
                milestone["checkpoint"] = ckpt_path.name
            milestone_f.write(json.dumps(milestone) + "\n"); milestone_f.flush()
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        "milestone/train_acc": m_train, "milestone/id_acc": m_id,
                        "milestone/ood_val_acc": m_val,
                        "milestone/worst_experiment_val_acc": m_worst_val,
                    }, step=ep)
                except Exception as e:
                    print(f"[wandb] milestone log skipped: {e}", flush=True)
            print(f"[{rid}] milestone ep{epoch_number} train {m_train:.4f} id {m_id:.4f} "
                  f"val {m_val:.4f} worst {m_worst_val:.4f}", flush=True)

    # ---------------- evaluation ----------------
    # STAGE GATE (PLAN.md): Stage 1 and 2 are decided on the OOD VALIDATION split only. The OOD
    # TEST split must stay untouched until the Stage-3 confirmatory runs, otherwise every
    # selection made from Stage-1 numbers is contaminated and the confirmatory test is no longer
    # confirmatory. Below stage 3 we therefore do not even compute test accuracy, and every
    # downstream metric (worst environment, mechanism, route reliance) is computed on OOD val.
    stage = int(cfg.get("stage", 1))
    # Terminal numbers must be read at the TARGET sparsity, never at an intermediate annealed k.
    if anneal_epochs:
        protocol["final_top_k"] = model.set_top_k(target_top_k)
    if bool(cfg.get("analysis", {}).get("record_train_accuracy", False)):
        acc_train, worst_train, _, _ = evaluate(
            model, train_loader, device, group_source=group_source)
    else:
        acc_train = worst_train = None
    acc_within, worst_within, _, _ = evaluate(
        model, test_within, device, group_source=group_source)

    if val_loader is None:
        # No OOD val split exists for this dataset: selection must fall back to the held-out
        # split, which is recorded explicitly so the contamination is visible in the JSON.
        sel_loader, selection_split = test_heldout, "ood_test(no_val_split)"
    else:
        sel_loader, selection_split = val_loader, "ood_val"

    acc_val, worst_val, per_env_val, per_env_n_val = evaluate(
        model, sel_loader, device, group_source=group_source)

    if stage >= 3:
        acc_ood, worst_ood, per_env_ood, per_env_n_ood = evaluate(
            model, test_heldout, device, group_source=group_source)
        test_evaluated = True
    else:
        acc_ood = worst_ood = None
        per_env_ood = per_env_n_ood = None
        test_evaluated = False
        print(f"[stage {stage}] OOD TEST not evaluated by design; selecting on {selection_split}",
              flush=True)

    # Everything mechanistic is measured on the split we are allowed to look at.
    mech_eval_loader = sel_loader
    acc_sel = acc_val

    # ---------------- mechanism ----------------
    mech = {}
    run_mechanism = bool(cfg.get("analysis", {}).get("run_mechanism", True))
    if run_mechanism and model.moe_blocks:
        per_block = {}
        for block_index, block in zip(cap.block_indices, model.moe_blocks):
            try:
                eidx, site, label = audit_routing.capture(
                    model, audit_loader, device, block=block)
                used, ent = audit_routing.expert_usage(eidx, cfg["model"]["n_experts"])
                per_block[str(block_index)] = {
                    "routing_mi_site": float(audit_routing.routing_mi(eidx, site)),
                    "routing_mi_class": float(audit_routing.routing_mi(eidx, label)),
                    "experts_used": float(used), "routing_entropy": float(ent),
                }
            except Exception as e:
                per_block[str(block_index)] = {"routing_error": str(e)}
        mech["routing_by_block"] = per_block
        # Preserve the historical flat fields for single-block aggregators.
        first = per_block.get(str(cap.block_indices[0]), {})
        for key in ("routing_mi_site", "routing_mi_class", "experts_used", "routing_entropy"):
            if key in first:
                mech[key] = first[key]
        try:
            mech["randomized_routes_acc"] = counterfactual_reroute(
                model, mech_eval_loader, device, seed=cfg["seed"],
                group_source=group_source)
            if mech["randomized_routes_acc"] is not None:
                mech["route_reliance"] = acc_sel - mech["randomized_routes_acc"]
        except Exception as e:
            mech["reroute_error"] = str(e)
        # Oracle arms have no randomisable router; shared residual arms additionally need this
        # correction-off ablation to separate routing from an independently useful shared path.
        try:
            shared_only = shared_only_accuracy(
                model, mech_eval_loader, device, group_source=group_source)
            if shared_only is not None:
                mech["shared_only_acc"] = shared_only
                mech["oracle_expert_contribution"] = acc_sel - shared_only
        except Exception as e:
            mech["shared_only_error"] = str(e)
        try:
            diversity = model.expert_diversity_loss()
            # Mean pairwise cosine between experts: ~1.0 means interchangeable experts, which is
            # the failure mode the balance loss cannot see. None = variant defines no measure.
            mech["expert_output_cosine"] = None if diversity is None else float(diversity)
        except Exception as e:
            mech["diversity_error"] = str(e)
    if run_mechanism:
        try:
            feats, site, label = audit_leak.features_site_label(model, audit_loader, device)
            mech["site_leakage"] = float(audit_leak.site_leakage(feats, site))
            mech["class_decodability"] = float(audit_leak.class_decodability(feats, label))
        except Exception as e:
            mech["leakage_error"] = str(e)
    router_deltas = router_parameter_deltas(model, initial_routers)

    sha, dirty = git_info()
    result = {
        "run_id": rid, "dataset": cfg["dataset"], "seed": cfg["seed"],
        # ---- the four grid factors + variant ----
        "variant": cap.variant, "placement": cap.placement, "routing_unit": cfg["model"]["routing_unit"],
        "geometry": cfg["model"]["geometry"], "pressure": pressure,
        "balance": cfg["model"]["balance"],
        "routing_estimator": cfg["model"].get("routing_estimator", "selected_st"),
        "classification_objective": objective,
        "cross_experiment_contrastive_w": consistency_w,
        "cross_experiment_pairs": paired_batches,
        "n_experts": cfg["model"]["n_experts"], "top_k": cfg["model"]["top_k"],
        # ---- outcomes ----
        # Stage <3: acc_heldout is null BY DESIGN (the OOD test split is not touched).
        # `acc_selection` is the number every Stage-1/2 decision is allowed to use.
        "stage": stage, "selection_split": selection_split, "test_evaluated": test_evaluated,
        "acc_selection": acc_sel,
        "acc_val": acc_val, "worst_env_val": worst_val,
        "per_env_val": per_env_val, "per_env_n_val": per_env_n_val,
        "acc_heldout": acc_ood, "worst_env_heldout": worst_ood,
        "per_env_heldout": per_env_ood, "per_env_n_heldout": per_env_n_ood,
        "acc_within": acc_within,
        "acc_train": acc_train, "worst_env_train": worst_train,
        "degradation_gap": (acc_within - acc_sel) if acc_sel is not None else None,
        "degradation_gap_test": (acc_within - acc_ood) if acc_ood is not None else None,
        # ---- resource accounting ----
        "total_params": cap.total_params, "ffn_block_params": cap.ffn_block_params,
        "router_params": cap.router_params, "active_ffn_params": cap.active_ffn_params,
        "training_total_params": (cap.total_params +
                                  (sum(p.numel() for p in adversary.parameters())
                                   if adversary is not None else 0)),
        "adversary_params": (sum(p.numel() for p in adversary.parameters())
                             if adversary is not None else 0),
        "block_index": cap.block_index, "block_indices": list(cap.block_indices),
        "n_blocks_converted": cap.n_converted_blocks,
        # ---- mechanism ----
        "initial_router_task_grad_norms": initial_router_task_grad_norms,
        # Per-epoch router gradient norms: shows WHEN the task signal to the router decays, which
        # a single initialisation probe cannot. Train accuracy saturates well before epoch 30.
        "router_grad_norm_by_epoch": router_grad_norm_by_epoch or None,
        "router_parameter_deltas": router_deltas,
        "group_dro_state": (group_dro.state() if group_dro is not None else None),
        "group_dro_max_weight_by_epoch": group_dro_weight_trace or None,
        **mech,
        # ---- provenance ----
        "protocol": protocol, "git_sha": sha, "git_dirty": dirty,
        "backbone_provenance": model.backbone_provenance,
        "tracking": ({"project": getattr(wandb_run, "project", None),
                      "group": getattr(wandb_run, "group", None),
                      "run_id": getattr(wandb_run, "id", None)}
                     if wandb_run is not None else None),
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "wall_seconds": round(time.time() - t0, 1),
        "config": cfg,
    }
    # Selection-stage runs never evaluate OOD test.  Keep every withheld field explicitly null
    # in the source result, rather than relying on the publication path to repair legacy empty
    # maps after the fact.  This makes the persisted JSON fail-closed even when HF publication is
    # disabled or interrupted.
    result = normalize_withheld_ood_fields(result)
    out_json.write_text(json.dumps(result, indent=2))
    logf.close()
    if milestone_f is not None:
        milestone_f.close()
    _test_str = f"{acc_ood:.4f}" if acc_ood is not None else "withheld"
    print(f"[done] {rid}  sel({selection_split}) {acc_sel:.4f}  within {acc_within:.4f}  "
          f"test {_test_str}  -> {out_json}")

    # ---- optional experiment trackers (never fatal) ----
    if wandb_run is not None:
        try:
            wandb_run.log({f"final/{k}": v for k, v in result.items()
                           if isinstance(v, (int, float))}, step=epochs)
            wandb_run.finish()
        except Exception as e:
            print(f"[wandb] skipped: {e}")
    if os.environ.get("HF_TOKEN") and os.environ.get("CCAS_HF_REPO"):
        try:
            validate_publishable_artifacts(result, milestone_path if milestones else None)
            artifacts = [out_json, log_path]
            if milestones:
                artifacts.append(milestone_path)
            artifacts.extend(sorted(out_dir.glob(f"{rid}.epoch*.pt")))
            published = publish_hf_run(result, artifacts, out_dir)
            print(f"[hf] uploaded validated folder {published['prefix']} "
                  f"({published['n_files']} files)")
        except Exception as e:
            print(f"[hf] skipped: {e}")


if __name__ == "__main__":
    main()
