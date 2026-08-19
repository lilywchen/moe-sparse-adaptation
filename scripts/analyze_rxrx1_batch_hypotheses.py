#!/usr/bin/env python
"""Quantify RxRx1 batch effects as a falsifiable H0--H4 ladder.

The default audit is test-blind: it extracts frozen Cell-DINO features from train, ID-test
(the second site of seen experiments), and OOD validation only.  It separates biology, additive
experiment shift, experiment-by-perturbation interaction, and site/image noise; evaluates
leave-one-experiment-out perturbation retrieval; estimates interaction rank; and asks whether a
small shared family of correction operators is predictable from unlabelled batch moments.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.model import build_ccas
from moe_shift.data import make_loaders, make_val_loader
from moe_shift.utils.config import load_config


def _extract(model, loader, device, split):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            feature = model.forward_features(batch[0].to(device)).float().cpu().numpy()
            for index in range(len(feature)):
                rows.append((feature[index], int(batch[1][index]), int(batch[3][index]),
                             int(batch[4][index]), split))
    return rows


def extract_cache(config, cache, include_test=False):
    cfg = load_config(config)
    cfg["model"]["variant"] = "original"
    cfg["model"]["batch_corrector"] = {"mode": "none"}
    cfg["train"]["experiment_batching"] = False
    cfg["train"]["num_workers"] = min(8, int(cfg["train"].get("num_workers", 8)))
    train, within, test, _audit = make_loaders(cfg)
    val = make_val_loader(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_ccas(cfg).to(device)
    rows = []
    for name, loader in (("train", train), ("id_test", within), ("val", val)):
        rows.extend(_extract(model, loader, device, name))
    if include_test:
        rows.extend(_extract(model, test, device, "test"))
    features = np.stack([row[0] for row in rows]).astype(np.float16)
    np.savez_compressed(
        cache, features=features,
        labels=np.asarray([row[1] for row in rows], dtype=np.int16),
        experiments=np.asarray([row[2] for row in rows], dtype=np.int16),
        cells=np.asarray([row[3] for row in rows], dtype=np.int8),
        splits=np.asarray([row[4] for row in rows]),
    )


def _cosine_correct(query, query_labels, reference, reference_labels, block=256):
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    correct = []
    for start in range(0, len(query), block):
        prediction = reference_labels[np.argmax(query[start:start + block] @ reference.T, axis=1)]
        correct.append(prediction == query_labels[start:start + block])
    return np.concatenate(correct) if correct else np.zeros(0, dtype=bool)


def _cosine_accuracy(query, query_labels, reference, reference_labels, block=256):
    return float(_cosine_correct(query, query_labels, reference, reference_labels, block).mean())


def _bootstrap_mean(values, rng, draws=1000):
    values = np.asarray(values, dtype=float)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "n_experiments": int(len(values))}


def _rank_for_fraction(singular_values, fraction):
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-12)
    return int(np.searchsorted(cumulative, fraction) + 1)


def _crossed_bootstrap(records, rng, draws=1000):
    """Pigeonhole bootstrap over experiments AND perturbations, stratified by cell type."""
    matrices = []
    for cell_value in sorted({record[0] for record in records}):
        subset = [record for record in records if record[0] == cell_value]
        environments = sorted({record[1] for record in subset})
        labels = sorted({record[2] for record in subset})
        environment_index = {value: index for index, value in enumerate(environments)}
        label_index = {value: index for index, value in enumerate(labels)}
        matrix = np.full((len(environments), len(labels)), np.nan)
        for _cell, environment, label, correct in subset:
            matrix[environment_index[environment], label_index[label]] = float(correct)
        matrices.append(matrix)
    samples = []
    for _ in range(draws):
        cell_means = []
        for matrix in matrices:
            rows = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            columns = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            cell_means.append(float(np.nanmean(matrix[np.ix_(rows, columns)])))
        samples.append(float(np.mean(cell_means)))
    observed = float(np.mean([np.nanmean(matrix) for matrix in matrices]))
    return {"cell_balanced_mean": observed,
            "crossed_experiment_perturbation_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "resample_sd": float(np.std(samples, ddof=1)),
            "resample_quantiles_05_50_95": np.quantile(samples, [0.05, 0.5, 0.95]).tolist(),
            "draws": draws}


def _rank_correlation(left, right):
    left = np.argsort(np.argsort(np.asarray(left, dtype=float)))
    right = np.argsort(np.argsort(np.asarray(right, dtype=float)))
    return float(np.corrcoef(left, right)[0, 1]) if len(left) > 2 else float("nan")


def analyze(cache, seed=0):
    payload = np.load(cache)
    x = payload["features"].astype(np.float32)
    y, env, cell, split = (payload[key] for key in ("labels", "experiments", "cells", "splits"))
    if np.any(split == "test"):
        test_policy = "explicitly_unsealed"
    else:
        test_policy = "sealed"

    # Average the two sites/images within each experiment x perturbation.  Remaining dispersion
    # is the cell/site-noise term; prototypes carry the crossed experiment x perturbation design.
    groups = defaultdict(list)
    for index, key in enumerate(zip(cell.tolist(), env.tolist(), y.tolist())):
        groups[tuple(map(int, key))].append(index)
    prototypes, labels, experiments, cells, partitions, noise_by_env = (
        [], [], [], [], [], defaultdict(list))
    for (c, e, label), indices in groups.items():
        values = x[indices]
        mean = values.mean(axis=0)
        prototypes.append(mean); labels.append(label); experiments.append(e); cells.append(c)
        partitions.append("val" if all(split[index] == "val" for index in indices) else "seen")
        noise_by_env[e].append(float(np.mean(np.sum(np.square(values - mean), axis=1))))
    p = np.stack(prototypes)
    labels = np.asarray(labels); experiments = np.asarray(experiments); cells = np.asarray(cells)
    partitions = np.asarray(partitions)

    rng = np.random.default_rng(seed)
    per_cell = {}
    all_retrieval = defaultdict(list)
    component_energy = defaultdict(list)
    operator_rows, descriptor_rows, operator_cells, operator_partitions = [], [], [], []
    interaction_rank90 = []
    difficulty_rows, correctness = [], defaultdict(list)

    for c in sorted(np.unique(cells)):
        mask_c = cells == c
        pc, yc, ec, sc = p[mask_c], labels[mask_c], experiments[mask_c], partitions[mask_c]
        envs = sorted(np.unique(ec))
        seen = sc == "seen"
        label_to_mean = {label: pc[seen & (yc == label)].mean(axis=0)
                         for label in np.unique(yc[seen])}
        grand = np.stack(list(label_to_mean.values())).mean(axis=0)
        biology = np.mean([np.sum(np.square(value - grand)) for value in label_to_mean.values()])
        additive_values, interaction_values = [], []
        ranks, raw_acc, centered_acc, adabn_acc = [], [], [], []
        cell_operators, cell_descriptors = [], []

        # A shared biological PCA basis makes A_e a small, comparable r x r operator.
        ordered_labels = sorted(label_to_mean)
        biology_matrix = np.stack([label_to_mean[label] - grand for label in ordered_labels])
        _u, _s, vt = np.linalg.svd(biology_matrix, full_matrices=False)
        r = min(32, len(vt), max(1, len(ordered_labels) - 1))
        basis = vt[:r].T

        for heldout in envs:
            target_mask = ec == heldout
            # Validation experiments are never allowed into the reference bank. For a seen
            # experiment, leave that experiment out of the seen bank as well.
            source_mask = (ec != heldout) & seen
            common = sorted(set(yc[target_mask]) & set(yc[source_mask]))
            target = np.stack([pc[target_mask & (yc == label)].mean(axis=0) for label in common])
            source = np.stack([pc[source_mask & (yc == label)].mean(axis=0) for label in common])
            common = np.asarray(common)
            raw_correct = _cosine_correct(target, common, source, common)
            raw_acc.append(float(raw_correct.mean()))
            target_centered = target - target.mean(axis=0)
            source_centered = source - source.mean(axis=0)
            center_correct = _cosine_correct(target_centered, common, source_centered, common)
            centered_acc.append(float(center_correct.mean()))
            target_scaled = target_centered / np.maximum(target.std(axis=0), 1e-4)
            source_scaled = source_centered / np.maximum(source.std(axis=0), 1e-4)
            adabn_correct = _cosine_correct(target_scaled, common, source_scaled, common)
            adabn_acc.append(float(adabn_correct.mean()))
            for label, raw_value, center_value, adabn_value in zip(
                    common, raw_correct, center_correct, adabn_correct):
                correctness["raw"].append((int(c), int(heldout), int(label), bool(raw_value)))
                correctness["center"].append(
                    (int(c), int(heldout), int(label), bool(center_value)))
                correctness["adabn"].append(
                    (int(c), int(heldout), int(label), bool(adabn_value)))

            mu = source
            residual = target - mu
            batch_shift = residual.mean(axis=0)
            interaction = residual - batch_shift
            additive_values.append(float(np.mean(np.sum(np.square(np.broadcast_to(
                batch_shift, residual.shape)), axis=1))))
            interaction_values.append(float(np.mean(np.sum(np.square(interaction), axis=1))))
            singular = np.linalg.svd(interaction, compute_uv=False)
            ranks.append(_rank_for_fraction(singular, 0.90))
            interaction_rank90.append(ranks[-1])
            additive_energy = additive_values[-1]
            interaction_energy = interaction_values[-1]
            difficulty_rows.append({
                "cell_type": int(c), "experiment": int(heldout),
                "n_perturbations": int(len(common)), "raw_retrieval": raw_acc[-1],
                "center_retrieval": centered_acc[-1], "adabn_retrieval": adabn_acc[-1],
                "predictive_harm": 1.0 - raw_acc[-1],
                "center_gain": centered_acc[-1] - raw_acc[-1],
                "adabn_gain": adabn_acc[-1] - raw_acc[-1],
                "additive_energy": additive_energy, "interaction_energy": interaction_energy,
                "interaction_fraction": interaction_energy /
                    max(additive_energy + interaction_energy, 1e-12),
                "interaction_rank90": ranks[-1],
                "site_noise": float(np.mean(noise_by_env[heldout]))
                    if noise_by_env[heldout] else float("nan"),
            })

            z = (mu - mu.mean(axis=0)) @ basis
            q = interaction @ basis
            ridge = 1e-2 * np.trace(z.T @ z) / max(r, 1)
            operator = np.linalg.solve(z.T @ z + ridge * np.eye(r), z.T @ q)
            cell_operators.append(operator.reshape(-1))
            # Unlabelled descriptor: experiment-wide first two moments only.
            cell_descriptors.append(np.concatenate((target.mean(axis=0),
                                                    np.log(target.std(axis=0) + 1e-4))))

        component_energy["biology"].append(float(biology))
        component_energy["additive"].extend(additive_values)
        component_energy["interaction"].extend(interaction_values)
        component_energy["site_noise"].extend(
            [float(np.mean(noise_by_env[e])) for e in envs if noise_by_env[e]])
        all_retrieval["raw"].extend(raw_acc)
        all_retrieval["center"].extend(centered_acc)
        all_retrieval["adabn"].extend(adabn_acc)
        per_cell[str(int(c))] = {
            "n_experiments": len(envs), "n_perturbations": len(ordered_labels),
            "retrieval_raw": float(np.mean(raw_acc)),
            "retrieval_center": float(np.mean(centered_acc)),
            "retrieval_adabn": float(np.mean(adabn_acc)),
            "median_interaction_rank90": float(np.median(ranks)),
        }
        operator_rows.extend(cell_operators)
        descriptor_rows.extend(cell_descriptors)
        operator_cells.extend([int(c)] * len(cell_operators))
        env_partition = {int(value): ("val" if np.all(sc[ec == value] == "val") else "seen")
                         for value in envs}
        operator_partitions.extend([env_partition[int(value)] for value in envs])

    operators = np.stack(operator_rows)
    descriptors = np.stack(descriptor_rows)
    operator_cells = np.asarray(operator_cells)
    operator_partitions = np.asarray(operator_partitions)
    shared_family, gate_r2, validation_gate_errors, validation_gate_denoms = {}, [], [], []
    for c in sorted(np.unique(operator_cells)):
        rows = operator_cells == c
        train_rows = rows & (operator_partitions == "seen")
        val_rows = rows & (operator_partitions == "val")
        a_raw = operators[train_rows]
        a_mean = a_raw.mean(axis=0)
        a = a_raw - a_mean
        d_raw = descriptors[train_rows]
        d_mean, d_std = d_raw.mean(axis=0), np.maximum(d_raw.std(axis=0), 1e-6)
        d = (d_raw - d_mean) / d_std
        u, s, _vt = np.linalg.svd(a, full_matrices=False)
        shared_family[str(int(c))] = {
            "top1_energy": float(np.square(s[:1]).sum() / max(np.square(s).sum(), 1e-12)),
            "top4_energy": float(np.square(s[:4]).sum() / max(np.square(s).sum(), 1e-12)),
            "rank90": _rank_for_fraction(s, 0.90),
        }
        k = min(4, len(s) - 1)
        if k < 1 or len(a) < 4:
            continue
        coefficients = u[:, :k] * s[:k]
        predictions = np.zeros_like(coefficients)
        # Fixed-alpha dual ridge, leave-one-experiment-out.  No perturbation labels enter the
        # descriptor, and alpha is not tuned on the held-out experiment.
        for index in range(len(d)):
            train = np.arange(len(d)) != index
            kernel = d[train] @ d[train].T / d.shape[1]
            cross = d[index] @ d[train].T / d.shape[1]
            predictions[index] = cross @ np.linalg.solve(
                kernel + np.eye(kernel.shape[0]), coefficients[train])
        denom = np.square(coefficients - coefficients.mean(axis=0)).sum()
        gate_r2.append(float(1.0 - np.square(coefficients - predictions).sum() / max(denom, 1e-12)))
        if np.any(val_rows):
            val_d = (descriptors[val_rows] - d_mean) / d_std
            val_coefficients = (operators[val_rows] - a_mean) @ _vt[:k].T
            kernel = d @ d.T / d.shape[1]
            cross = val_d @ d.T / d.shape[1]
            val_predictions = cross @ np.linalg.solve(
                kernel + np.eye(kernel.shape[0]), coefficients)
            validation_gate_errors.append(float(np.square(
                val_coefficients - val_predictions).sum()))
            validation_gate_denoms.append(float(np.square(val_coefficients).sum()))

    energy_total = sum(np.mean(values) for values in component_energy.values())
    energy = {key: {"mean_squared_norm": float(np.mean(values)),
                    "fraction": float(np.mean(values) / max(energy_total, 1e-12))}
              for key, values in component_energy.items()}
    retrieval = {key: _bootstrap_mean(values, rng) for key, values in all_retrieval.items()}
    crossed_retrieval = {key: _crossed_bootstrap(values, rng)
                         for key, values in correctness.items()}
    difficulty_arrays = {key: np.asarray([row[key] for row in difficulty_rows], dtype=float)
                         for key in ("predictive_harm", "center_gain", "adabn_gain",
                                     "additive_energy", "interaction_energy",
                                     "interaction_fraction", "interaction_rank90", "site_noise")}
    difficulty_distribution = {
        key: {"mean": float(np.nanmean(values)),
              "quantiles_05_25_50_75_95": np.nanquantile(
                  values, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()}
        for key, values in difficulty_arrays.items()}
    difficulty_correlations = {
        key: _rank_correlation(difficulty_arrays["predictive_harm"], values)
        for key, values in difficulty_arrays.items() if key != "predictive_harm"}
    hardest = sorted(difficulty_rows, key=lambda row: row["predictive_harm"], reverse=True)[:8]
    perturbation_scores = defaultdict(list)
    for cell_value, _environment, label, correct in correctness["raw"]:
        perturbation_scores[(cell_value, label)].append(float(correct))
    perturbation_difficulty = np.asarray(
        [1.0 - np.mean(values) for values in perturbation_scores.values()], dtype=float)
    med_rank = float(np.median(interaction_rank90))
    top4 = float(np.mean([value["top4_energy"] for value in shared_family.values()]))
    mean_gate_r2 = float(np.mean(gate_r2)) if gate_r2 else float("nan")
    validation_gate_r2 = (float(1.0 - sum(validation_gate_errors) /
                                max(sum(validation_gate_denoms), 1e-12))
                          if validation_gate_errors else float("nan"))
    conclusions = {
        "H0_no_harmful_batch": bool(retrieval["raw"]["mean"] >= 0.95),
        "H1_additive_sufficient": bool(
            retrieval["center"]["mean"] - retrieval["raw"]["mean"] >= 0.02
            and energy["interaction"]["fraction"] < 0.10),
        "H2_affine_supported": bool(
            retrieval["adabn"]["mean"] - retrieval["center"]["mean"] >= 0.02),
        "H3_low_rank_supported": bool(med_rank <= 8),
        "H4_shared_family_supported": bool(top4 >= 0.80 and mean_gate_r2 > 0.10),
    }
    return {
        "schema_version": 1, "cache": str(Path(cache).resolve()), "test_policy": test_policy,
        "estimand": "class-matched Cell-DINO feature geometry within cell type",
        "n_images": int(len(x)), "n_experiments": int(len(np.unique(env))),
        "n_prototypes": int(len(p)), "component_energy": energy,
        "retrieval_leave_one_experiment_out": retrieval,
        "retrieval_crossed_resampling": crossed_retrieval,
        "batch_difficulty_distribution": difficulty_distribution,
        "batch_difficulty_rank_correlations": difficulty_correlations,
        "per_experiment_difficulty": difficulty_rows,
        "hardest_experiments": hardest,
        "perturbation_difficulty_distribution": {
            "n_cell_perturbations": int(len(perturbation_difficulty)),
            "quantiles_05_25_50_75_95": np.quantile(
                perturbation_difficulty, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()},
        "interaction_rank90_median": med_rank, "shared_operator_family": shared_family,
        "unlabelled_descriptor_to_operator_coefficient_loo_r2_mean": mean_gate_r2,
        "unlabelled_descriptor_to_operator_coefficient_heldout_val_r2": validation_gate_r2,
        "per_cell_type": per_cell, "predeclared_decision_thresholds": {
            "H0": "raw retrieval >= .95", "H1": "centering gain >= .02 and interaction <10%",
            "H2": "AdaBN gain over centering >= .02", "H3": "median rank90 <= 8",
            "H4": "top-4 operator-family energy >= .80 and descriptor LOO R2 > .10",
        }, "hypotheses_supported": conclusions,
    }


def render_markdown(result):
    lines = ["# RxRx1 batch-effect hypothesis audit", "",
             f"Test partition: **{result['test_policy']}**  ",
             f"Images: {result['n_images']:,}; experiments: {result['n_experiments']}; "
             f"experiment×perturbation prototypes: {result['n_prototypes']:,}", "",
             "## Variance decomposition", "",
             "| Component | Energy fraction |", "|---|---:|"]
    for key, value in result["component_energy"].items():
        lines.append(f"| {key} | {100 * value['fraction']:.2f}% |")
    lines += ["", "## Leave-one-experiment-out retrieval", "",
              "| Correction | Accuracy | Experiment-bootstrap 95% CI |", "|---|---:|---:|"]
    for key, value in result["retrieval_leave_one_experiment_out"].items():
        lo, hi = value["ci95"]
        lines.append(f"| {key} | {100*value['mean']:.2f}% | {100*lo:.2f}–{100*hi:.2f}% |")
    lines += ["", "Crossed experiment×perturbation resampling (cell-balanced):", "",
              "| Correction | Mean | Crossed 95% interval | Resample SD |", "|---|---:|---:|---:|"]
    for key, value in result["retrieval_crossed_resampling"].items():
        lo, hi = value["crossed_experiment_perturbation_ci95"]
        lines.append(f"| {key} | {100*value['cell_balanced_mean']:.2f}% | "
                     f"{100*lo:.2f}–{100*hi:.2f}% | {100*value['resample_sd']:.2f} |")
    lines += ["", "## Batch difficulty distribution", "",
              "| Quantity | 5% | 25% | median | 75% | 95% |", "|---|---:|---:|---:|---:|---:|"]
    for key, value in result["batch_difficulty_distribution"].items():
        quantiles = value["quantiles_05_25_50_75_95"]
        lines.append("| " + key + " | " + " | ".join(f"{number:.4f}" for number in quantiles) + " |")
    lines += ["", "Hardest experiments by class-matched predictive harm:", ""]
    for row in result["hardest_experiments"][:5]:
        lines.append(f"- experiment {row['experiment']} (cell {row['cell_type']}): "
                     f"harm={row['predictive_harm']:.3f}, affine gain={row['adabn_gain']:+.3f}, "
                     f"interaction fraction={row['interaction_fraction']:.3f}, "
                     f"rank90={row['interaction_rank90']}")
    lines += ["", "## Falsification ladder", ""]
    for key, supported in result["hypotheses_supported"].items():
        lines.append(f"- {key}: **{'supported' if supported else 'rejected'}**")
    lines += ["", f"Median interaction rank for 90% energy: "
              f"{result['interaction_rank90_median']:.1f}  ",
              "Unlabelled moment descriptor → shared-operator coefficient LOO R²: "
              f"{result['unlabelled_descriptor_to_operator_coefficient_loo_r2_mean']:.3f}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ccas_rxrx1_cell_dino_native.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-test", action="store_true",
                        help="explicitly unseal test (off by default)")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    cache, output = Path(args.cache), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.extract or not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        extract_cache(args.config, cache, include_test=args.include_test)
    result = analyze(cache)
    (output / "batch_hypotheses.json").write_text(json.dumps(result, indent=2))
    (output / "batch_hypotheses.md").write_text(render_markdown(result))
    print(render_markdown(result))


if __name__ == "__main__":
    main()
