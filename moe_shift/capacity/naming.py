"""Run identity — deliberately torch-free so sweeps/aggregators can plan without importing torch.

The run_id encodes every factor that changes the experiment, so results are addressable and
sweeps are idempotent (a cell whose JSON exists is skipped).
"""

import hashlib


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
    corrector = m.get("batch_corrector", {})
    corrector_mode = str(corrector.get("mode", "none"))
    if corrector_mode != "none":
        fragment = f"bc-{corrector_mode}"
        if corrector_mode in ("lowrank", "moe_batch", "moe_dual"):
            fragment += f"-E{int(corrector.get('n_experts', 1))}r{int(corrector.get('rank', 16))}"
        parts.append(fragment)
    if v in ("moe", "moe_frozen", "shared_moe"):
        # Prefix the estimator to the historical identity. This prevents result-file collisions
        # while leaving the established dataset/variant/campaign substrings usable as anchor keys.
        parts.insert(0, str(m.get("routing_estimator", "selected_st")))
        pressure = m.get("pressure", "route" if m.get("balance") in
                         ("within_batch", "within_environment") else "canonical")
        bank = (f"S1E{m['n_experts']}k{m['top_k']}" if v == "shared_moe"
                else f"E{m['n_experts']}k{m['top_k']}")
        parts += [location, m["routing_unit"], m["geometry"], pressure, bank]
    elif v == "dense_wide":
        parts += [location, m.get("pressure", "canonical"), f"E{m['n_experts']}"]
    elif v in FRONTIER_VARIANTS:
        # Every factor that changes a frontier arm's function must appear here, or two distinct
        # arms would collide on one result file and the sweep would silently skip the second.
        parts += [location, *_frontier_identity(v, m)]
    parts += [f"ep{t['epochs']}", f"s{cfg['seed']}"]
    # A BTX-mixed model and a plain shared_moe model share every other factor, so without this
    # marker they would collide on one result file and the sweep would skip the second silently.
    if m.get("btx_manifest"):
        parts.append("btx" if m.get("btx_freeze_experts", True) else "btxopen")
    if t.get("environment_subset"):
        environments = sorted(int(value) for value in t["environment_subset"])
        digest = hashlib.sha256(",".join(map(str, environments)).encode()).hexdigest()[:8]
        parts.append(f"envsub{len(environments)}-{digest}")
    if str(t.get("objective", "erm")) == "group_dro" or t.get("group_dro"):
        parts.append("dro")
    if float(t.get("label_smoothing", 0.0) or 0.0) > 0:
        parts.append(f"ls{_short(t['label_smoothing'])}")
    if t.get("anneal_top_k_epochs"):
        parts.append(f"anneal{int(t['anneal_top_k_epochs'])}")
    if cfg.get("run_tag"):
        parts.append(str(cfg["run_tag"]))
    return "_".join(parts)


FRONTIER_VARIANTS = ("oracle_moe", "condln_moe", "soft_moe", "lowrank_moe")


def _short(value) -> str:
    """Compact numeric tag: 0.05 -> '005', 16 -> '16'. Keeps run ids filesystem-friendly."""
    number = float(value)
    if number == int(number):
        return str(int(number))
    return ("%g" % number).replace("0.", "0").replace(".", "")


def _frontier_identity(variant, model_cfg):
    """Identity fragments for a frontier variant (torch-free, so sweeps can plan without torch)."""
    m = model_cfg
    if variant == "oracle_moe":
        return [
            f"oracle-{m.get('group_source', 'cell_type')}",
            f"E{m['n_experts']}",
            f"drop{_short(m.get('expert_dropout', 0.5))}",
        ]
    if variant == "condln_moe":
        return [
            f"condln-{m.get('condln_descriptor', 'token_stats')}",
            str(m.get("condln_modulate", "input")),
            m["geometry"],
            f"E{m['n_experts']}k{m['top_k']}",
        ]
    if variant == "soft_moe":
        rank = int(m.get("expert_rank", 0) or 0)
        return [
            "soft",
            f"S{m.get('slots_per_expert', 1)}E{m['n_experts']}",
            (f"r{rank}" if rank > 0 else "rfull"),
            f"t{_short(m.get('soft_temperature', 1.0))}",
        ]
    if variant == "lowrank_moe":
        fragments = [
            m["routing_unit"], m["geometry"],
            f"E{m['n_experts']}k{m['top_k']}r{int(m.get('expert_rank', 16))}",
        ]
        if float(m.get("diversity_w", 0.0) or 0.0) > 0:
            fragments.append(f"div{_short(m['diversity_w'])}")
        return fragments
    raise ValueError(f"no identity rule for frontier variant {variant!r}")
