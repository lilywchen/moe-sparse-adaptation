"""Run identity — deliberately torch-free so sweeps/aggregators can plan without importing torch.

The run_id encodes every factor that changes the experiment, so results are addressable and
sweeps are idempotent (a cell whose JSON exists is skipped).
"""


def explicit_block_indices(model_cfg):
    """Return the canonical explicit FFN indices, accepting the public descriptive alias.

    ``block_indices`` is the original internal key.  The fast-discovery protocol introduced the
    clearer ``ffn_block_indices`` name.  Both must drive model construction *and* run identity;
    accepting the alias in only one place can silently label a legacy-placement model as a
    conflict-localized intervention.
    """
    legacy = model_cfg.get("block_indices")
    explicit = model_cfg.get("ffn_block_indices")
    if legacy is not None and explicit is not None:
        legacy_values = [int(i) for i in legacy]
        explicit_values = [int(i) for i in explicit]
        if legacy_values != explicit_values:
            raise ValueError(
                "model.block_indices and model.ffn_block_indices disagree: "
                f"{legacy_values} != {explicit_values}"
            )
        return explicit_values
    values = explicit if explicit is not None else legacy
    return None if values is None else [int(i) for i in values]


def run_id_from(cfg) -> str:
    m, t = cfg["model"], cfg["train"]
    v = m["variant"]
    block_indices = explicit_block_indices(m)
    if block_indices is not None:
        location = "blocks" + "-".join(str(i) for i in block_indices)
    elif m.get("placements") is not None:
        location = "+".join(str(p) for p in m["placements"])
    else:
        location = m["placement"]
    parts = [cfg["dataset"], v]
    if v in ("moe", "moe_frozen"):
        # Prefix the estimator to the historical identity. This prevents result-file collisions
        # while leaving the established dataset/variant/campaign substrings usable as anchor keys.
        parts.insert(0, str(m.get("routing_estimator", "selected_st")))
        pressure = m.get("pressure", "route" if m.get("balance") in
                         ("within_batch", "within_environment") else "canonical")
        parts += [location, m["routing_unit"], m["geometry"], pressure,
                  f"E{m['n_experts']}k{m['top_k']}"]
    elif v == "dense_wide":
        parts += [location, m.get("pressure", "canonical"), f"E{m['n_experts']}"]
    parts += [f"ep{t['epochs']}", f"s{cfg['seed']}"]
    if cfg.get("run_tag"):
        parts.append(str(cfg["run_tag"]))
    return "_".join(parts)
