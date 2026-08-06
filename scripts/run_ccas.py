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


def classification_objective(logits, labels, environments, objective="erm"):
    """Return the supervised loss while making environment weighting explicit.

    ``environment_balanced`` gives every experiment represented in the minibatch equal weight,
    rather than letting experiments with more images dominate the update.  It uses no validation
    or test information and changes no model capacity.
    """
    per_example = F.cross_entropy(logits, labels, reduction="none")
    if objective == "erm":
        return per_example.mean()
    if objective != "environment_balanced":
        raise ValueError(f"unknown train.objective: {objective!r}")
    present = torch.unique(environments)
    if torch.any(present < 0):
        raise ValueError("environment_balanced training requires non-negative train environments")
    return torch.stack([per_example[environments == env].mean() for env in present]).mean()


def router_gradient_norms(loss, model):
    """Classification-loss gradient norm for each trainable router, without accumulating grads.

    This is a fail-fast audit for hard top-1 routing. A selected-ST run must expose a finite,
    nonzero task gradient before auxiliary losses are added; the historical renormalised top-1
    implementation did not. Frozen-router and non-MoE models correctly return an empty mapping.
    """
    norms = {}
    for block_index, block in zip(model.capacity.block_indices, model.moe_blocks):
        params = [p for p in block.router.parameters() if p.requires_grad]
        if not params:
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        squared = loss.new_zeros(())
        for grad in grads:
            if grad is not None:
                squared = squared + grad.detach().float().square().sum()
        norms[str(block_index)] = float(squared.sqrt())
    return norms


def snapshot_routers(model):
    """Small CPU snapshots used to prove that trainable routers actually moved."""
    return {
        str(block_index): [p.detach().float().cpu().clone() for p in block.router.parameters()]
        for block_index, block in zip(model.capacity.block_indices, model.moe_blocks)
    }


def router_parameter_deltas(model, initial):
    """Absolute and relative L2 displacement of every router from initialisation."""
    deltas = {}
    for block_index, block in zip(model.capacity.block_indices, model.moe_blocks):
        key = str(block_index)
        before = initial.get(key)
        if before is None:
            continue
        delta_sq = base_sq = 0.0
        for current, start in zip(block.router.parameters(), before):
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


@torch.no_grad()
def evaluate(model, loader, device):
    """-> (acc, worst_env_acc, per_env_acc, per_env_n), bucketed by RAW environment id.

    `per_env_n` is required, not cosmetic: the plan's uncertainty is a CLUSTER bootstrap over
    experiments / hospitals, which cannot be reconstructed from per-environment accuracies alone.
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
def counterfactual_reroute(model, loader, device, seed=0):
    """Joint route reliance with every converted router randomized and expert weights fixed.

    The router is temporarily replaced by i.i.d. Gaussian logits, which give a balanced random
    top-1 assignment while leaving every expert's weights untouched. The substitution is always
    undone (finally), and the RNG is seeded so the number is reproducible across reruns.
    """
    blocks = list(model.moe_blocks)
    if not blocks:
        return None
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    real_router_forwards = [block.router.forward for block in blocks]

    def randomized_forward(block):
        def forward(z):
            r = torch.randn(z.shape[0], block.n_experts, generator=gen)
            return r.to(z.device, dtype=z.dtype)
        return forward

    try:
        for block in blocks:
            block.router.forward = randomized_forward(block)
        acc = evaluate(model, loader, device)[0]
    finally:
        for block, real_forward in zip(blocks, real_router_forwards):
            block.router.forward = real_forward
    return acc


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

    initial_router_task_grad_norms = None
    for ep in range(epochs):
        model.train()
        if adversary is not None:
            adversary.train()
        run_loss = run_aux = run_adv = 0.0; nb = 0
        for batch in train_loader:
            x, y, s = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            model.set_env(s)                       # used ONLY by within-environment balancing
            feats = model.forward_features(x)
            logits = model.fc(feats)
            loss = classification_objective(logits, y, s, objective)
            if initial_router_task_grad_norms is None and model.moe_blocks:
                initial_router_task_grad_norms = router_gradient_norms(loss, model)
                print(
                    "[router-gradient-check] classification-only norms "
                    f"{initial_router_task_grad_norms}", flush=True)
            aux = model.aux_loss(bw, zw)
            adv = loss.new_zeros(())
            if adversary is not None:
                lambd = lambda_schedule(ep, epochs, inv_target)
                adv = F.cross_entropy(adversary(feats, lambd), s)
            (loss + aux + adv).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            run_loss += loss.item(); run_aux += float(aux); run_adv += float(adv); nb += 1
        sched.step()
        rec = {"epoch": ep, "loss": run_loss / max(nb, 1), "aux": run_aux / max(nb, 1),
               "adversary": run_adv / max(nb, 1),
               "lr": opt.param_groups[-1]["lr"], "t": round(time.time() - t0, 1)}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if wandb_run is not None:
            try:
                wandb_run.log({f"train/{k}": v for k, v in rec.items() if k != "epoch"},
                              step=ep)
            except Exception as e:
                print(f"[wandb] epoch log skipped: {e}", flush=True)
        print(f"[{rid}] ep{ep} loss {rec['loss']:.4f} aux {rec['aux']:.4f} "
              f"adv {rec['adversary']:.4f}", flush=True)

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
                m_train, m_worst_train, _, _ = evaluate(model, train_loader, device)
                m_id, m_worst_id, _, _ = evaluate(model, test_within, device)
                m_val, m_worst_val, m_per_env, m_per_env_n = evaluate(model, val_loader, device)
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
    if bool(cfg.get("analysis", {}).get("record_train_accuracy", False)):
        acc_train, worst_train, _, _ = evaluate(model, train_loader, device)
    else:
        acc_train = worst_train = None
    acc_within, worst_within, _, _ = evaluate(model, test_within, device)

    if val_loader is None:
        # No OOD val split exists for this dataset: selection must fall back to the held-out
        # split, which is recorded explicitly so the contamination is visible in the JSON.
        sel_loader, selection_split = test_heldout, "ood_test(no_val_split)"
    else:
        sel_loader, selection_split = val_loader, "ood_val"

    acc_val, worst_val, per_env_val, per_env_n_val = evaluate(model, sel_loader, device)

    if stage >= 3:
        acc_ood, worst_ood, per_env_ood, per_env_n_ood = evaluate(model, test_heldout, device)
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
                model, mech_eval_loader, device, seed=cfg["seed"])
            if mech["randomized_routes_acc"] is not None:
                mech["route_reliance"] = acc_sel - mech["randomized_routes_acc"]
        except Exception as e:
            mech["reroute_error"] = str(e)
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
        "router_parameter_deltas": router_deltas,
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
            validate_stage1_artifacts(result, milestone_path if milestones else None)
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
