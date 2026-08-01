#!/usr/bin/env python
"""One CCAS run: fixed-parameter-budget dense-vs-sparse adaptation under acquisition shift.

Writes a flat JSON containing everything needed to reproduce and audit the run: the full config,
git SHA + dirty flag, exact parameter accounting (total / router / active) with the budget delta
against the fixed reference P*, protocol assertions, all metrics, and the environment.

    python scripts/run_ccas.py --config configs/ccas_rxrx1.yaml \
        --override model.variant=moe model.placement=middle model.routing_unit=token \
                   model.geometry=cosine model.pressure=canonical seed=0
"""
import argparse, json, os, platform, socket, subprocess, sys, time
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
    """route reliance = OOD(learned routes) - OOD(randomized routes), expert weights fixed.

    The router is temporarily replaced by i.i.d. Gaussian logits, which give a balanced random
    top-1 assignment while leaving every expert's weights untouched. The substitution is always
    undone (finally), and the RNG is seeded so the number is reproducible across reruns.
    """
    blk = model.moe_block
    if blk is None:
        return None
    real_router_forward = blk.router.forward
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def rand_router(z):
        r = torch.randn(z.shape[0], blk.n_experts, generator=gen)
        return r.to(z.device, dtype=z.dtype)

    try:
        blk.router.forward = rand_router
        acc = evaluate(model, loader, device)[0]
    finally:
        blk.router.forward = real_router_forward
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
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
                group=f"{cfg['dataset']}_{cap.variant}", job_type="train",
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
    protocol["n_blocks"] = len(model.blocks)
    protocol["exactly_one_block_converted"] = True
    protocol["training_pressure"] = pressure
    protocol["route_balance"] = expected_balance
    protocol["output_adversary"] = adversary is not None
    if cap.variant in ("moe", "moe_frozen"):
        protocol["experts_are_upcycled_copies"] = True
        protocol["router_trainable"] = (cap.variant == "moe")

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
            loss = F.cross_entropy(logits, y)
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
        per_env_ood, per_env_n_ood = {}, {}
        test_evaluated = False
        print(f"[stage {stage}] OOD TEST not evaluated by design; selecting on {selection_split}",
              flush=True)

    # Everything mechanistic is measured on the split we are allowed to look at.
    mech_eval_loader = sel_loader
    acc_sel = acc_val

    # ---------------- mechanism ----------------
    mech = {}
    run_mechanism = bool(cfg.get("analysis", {}).get("run_mechanism", True))
    if run_mechanism and model.moe_block is not None:
        try:
            eidx, site, label = audit_routing.capture(model, audit_loader, device)
            mech["routing_mi_site"] = float(audit_routing.routing_mi(eidx, site))
            mech["routing_mi_class"] = float(audit_routing.routing_mi(eidx, label))
            used, ent = audit_routing.expert_usage(eidx, cfg["model"]["n_experts"])
            mech["experts_used"], mech["routing_entropy"] = float(used), float(ent)
        except Exception as e:
            mech["routing_error"] = str(e)
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

    sha, dirty = git_info()
    result = {
        "run_id": rid, "dataset": cfg["dataset"], "seed": cfg["seed"],
        # ---- the four grid factors + variant ----
        "variant": cap.variant, "placement": cap.placement, "routing_unit": cfg["model"]["routing_unit"],
        "geometry": cfg["model"]["geometry"], "pressure": pressure,
        "balance": cfg["model"]["balance"],
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
        "block_index": cap.block_index,
        # ---- mechanism ----
        **mech,
        # ---- provenance ----
        "protocol": protocol, "git_sha": sha, "git_dirty": dirty,
        "backbone_provenance": model.backbone_provenance,
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "wall_seconds": round(time.time() - t0, 1),
        "config": cfg,
    }
    out_json.write_text(json.dumps(result, indent=2))
    logf.close()
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
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.upload_file(path_or_fileobj=str(out_json), path_in_repo=f"results/{rid}.json",
                            repo_id=os.environ["CCAS_HF_REPO"], repo_type="dataset")
            print(f"[hf] uploaded results/{rid}.json")
        except Exception as e:
            print(f"[hf] skipped: {e}")


if __name__ == "__main__":
    main()
