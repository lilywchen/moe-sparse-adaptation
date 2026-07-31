"""Run identity — deliberately torch-free so sweeps/aggregators can plan without importing torch.

The run_id encodes every factor that changes the experiment, so results are addressable and
sweeps are idempotent (a cell whose JSON exists is skipped).
"""


def run_id_from(cfg) -> str:
    m, t = cfg["model"], cfg["train"]
    v = m["variant"]
    parts = [cfg["dataset"], v]
    if v in ("moe", "moe_frozen"):
        pressure = m.get("pressure", "route" if m.get("balance") in
                         ("within_batch", "within_environment") else "canonical")
        parts += [m["placement"], m["routing_unit"], m["geometry"], pressure,
                  f"E{m['n_experts']}k{m['top_k']}"]
    elif v == "dense_wide":
        parts += [m["placement"], m.get("pressure", "canonical"), f"E{m['n_experts']}"]
    parts += [f"ep{t['epochs']}", f"s{cfg['seed']}"]
    if cfg.get("run_tag"):
        parts.append(str(cfg["run_tag"]))
    return "_".join(parts)
