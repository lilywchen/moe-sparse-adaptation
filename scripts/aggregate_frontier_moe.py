#!/usr/bin/env python
"""One-command aggregator for the frontier-MoE wave. Runs from ANY container: shared storage.

Emits four things, in the order they should be read:

1. **Ceilings first.**  ``oracle_cell_type`` and ``oracle_environment`` bound what any learned
   router could achieve.  If neither clears the dense reference, the deployable arms below are
   explained and the honest conclusion is available immediately.
2. **The terminal table**, with the routing diagnostics beside the accuracies.  The previous wave
   produced accuracy alone and could not distinguish "the MoE helped" from "the extra parameters
   helped", which is the whole question.
3. **Paired contrasts** against the predeclared references, so no ranking is invented from the
   test split after the fact.
4. **Mechanism gates.**  ``route_reliance >= 0.01`` is the predeclared threshold from the
   completed campaigns, where the maximum ever observed was ``0.0065``.  ``expert_output_cosine``
   near 1.0 means the experts are interchangeable regardless of how balanced their usage is --
   the failure the canonical balance loss cannot see.

Nothing here selects a configuration.  OOD validation remains the only selection metric; OOD test
is a descriptive readout for this fixed, predefined set of arms.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Predeclared route-reliance gate. The completed campaigns never exceeded 0.0065 against it.
RELIANCE_GATE = 0.01

#: Historical 30-epoch references under the previous wave's code, for orientation only. These were
#: produced by a different commit, so in-wave contrasts are the trustworthy ones and these are
#: labelled as external.
EXTERNAL_REFERENCES = {
    "shared_E3k1_late2 (prior wave)": {
        "acc_val": 0.22224, "acc_heldout": 0.38877,
        "worst_env_heldout": 0.08361, "acc_within": 0.55380,
    },
    "dense expansion 10-11 (prior wave)": {
        "acc_val": 0.21514, "acc_heldout": 0.38731,
        "worst_env_heldout": 0.07951, "acc_within": None,
    },
    "original Cell-DINO (prior wave)": {
        "acc_val": 0.20154, "acc_heldout": 0.36524,
        "worst_env_heldout": 0.06352, "acc_within": None,
    },
}

CEILING_ARMS = ("oracle_cell_type", "oracle_environment")

METRICS = ("acc_val", "acc_heldout", "worst_env_heldout", "acc_within", "acc_train")


def load_wave(root):
    root = Path(root).expanduser().resolve()
    manifest_path = root / "wave_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no wave_manifest.json under {root}")
    manifest = json.loads(manifest_path.read_text())
    results = {}
    for entry in manifest["runs"]:
        path = root / f"{entry['run_id']}.json"
        if path.is_file():
            results[entry["label"]] = json.loads(path.read_text())
    return root, manifest, results


def _pct(value, width=8):
    if value is None:
        return "-".rjust(width)
    return f"{100.0 * float(value):.3f}%".rjust(width)


def _num(value, width=8, places=4):
    if value is None:
        return "-".rjust(width)
    return f"{float(value):.{places}f}".rjust(width)


def _delta(a, b, width=8):
    if a is None or b is None:
        return "-".rjust(width)
    return f"{100.0 * (float(a) - float(b)):+.3f}".rjust(width)


def validate(results):
    """Protocol checks that must pass BEFORE any arm is ranked."""
    problems = []
    for label, result in sorted(results.items()):
        protocol = result.get("protocol", {})
        if not result.get("test_evaluated"):
            problems.append(f"{label}: test_evaluated is false; no terminal readout")
        if result.get("selection_split") != "ood_val":
            problems.append(f"{label}: selection_split={result.get('selection_split')!r}")
        for key in ("acc_val", "acc_heldout"):
            value = result.get(key)
            if value is None or not (0.0 <= float(value) <= 1.0):
                problems.append(f"{label}: {key}={value!r} is not a valid accuracy")
        if "route_reliance" not in result and "randomized_routes_acc" not in result \
                and "shared_only_acc" not in result:
            problems.append(
                f"{label}: no routing counterfactual recorded "
                "(analysis.run_mechanism was probably false)")
        if protocol.get("trained_on_environment_subset"):
            problems.append(
                f"{label}: trained on an environment subset; not comparable with full-data arms")
        if not protocol.get("pretrained_shared_expert_always_active", True):
            problems.append(f"{label}: does not keep the pretrained dense FFN active")
    return problems


def render_ceilings(results):
    lines = ["", "== 1. CEILINGS (read these first) =========================================="]
    dense = EXTERNAL_REFERENCES["dense expansion 10-11 (prior wave)"]
    for label in CEILING_ARMS:
        result = results.get(label)
        if result is None:
            lines.append(f"  {label}: not complete")
            continue
        shared_only = result.get("shared_only_acc")
        contribution = result.get("oracle_expert_contribution")
        lines.append(
            f"  {label}: OOD val {_pct(result.get('acc_val'))}  "
            f"test {_pct(result.get('acc_heldout'))}  "
            f"shared-only {_pct(shared_only)}  "
            f"expert contribution {_num(contribution, places=4)}")
        gap = None
        if result.get("acc_val") is not None and dense["acc_val"] is not None:
            gap = float(result["acc_val"]) - dense["acc_val"]
        lines.append(
            f"      vs dense expansion on OOD val: {_delta(result.get('acc_val'), dense['acc_val'])} pts"
            + ("  -> ceiling clears dense" if gap and gap > 0 else
               "  -> ceiling does NOT clear dense"))
    lines.append(
        "  Decision rule (moe_shift/models/oracle.py): if a shared path trained alongside\n"
        "  ground-truth-indexed experts still cannot beat the parameter-matched dense control,\n"
        "  batch and content are not separable here even with ground truth, and no learned\n"
        "  router can rescue it.")
    return lines


def render_table(manifest, results):
    lines = ["", "== 2. TERMINAL TABLE ======================================================",
             "| Arm | Variant | OOD val | OOD test | Worst test | ID | Train | Reliance "
             "| Experts | Entropy | ExpCos | MI site |",
             "|" + "---|" * 12]
    order = [entry["label"] for entry in manifest["runs"]]
    for label in order:
        result = results.get(label)
        if result is None:
            lines.append(f"| {label} | - | " + " | ".join(["-"] * 10) + " |")
            continue
        lines.append(
            f"| {label} | {result.get('variant')} "
            f"| {_pct(result.get('acc_val'))} | {_pct(result.get('acc_heldout'))} "
            f"| {_pct(result.get('worst_env_heldout'))} | {_pct(result.get('acc_within'))} "
            f"| {_pct(result.get('acc_train'))} "
            f"| {_num(result.get('route_reliance'))} "
            f"| {_num(result.get('experts_used'), places=1)} "
            f"| {_num(result.get('routing_entropy'), places=3)} "
            f"| {_num(result.get('expert_output_cosine'), places=3)} "
            f"| {_num(result.get('routing_mi_site'), places=3)} |")
    lines.append("")
    lines.append("External 30-epoch references (previous commit; orientation only):")
    for label, reference in EXTERNAL_REFERENCES.items():
        lines.append(
            f"  {label}: val {_pct(reference['acc_val'])}  test {_pct(reference['acc_heldout'])}  "
            f"worst {_pct(reference['worst_env_heldout'])}")
    return lines


#: Default paired reference. None of the eight arms is a plain shared/residual MoE -- every one is
#: an intervention -- so contrasting them against each other would compare two changes at once.
#: The previous wave's best arm is therefore the default reference, labelled as external.
DEFAULT_REFERENCE = "shared_E3k1_late2 (prior wave)"


def render_contrasts(results, reference_label=DEFAULT_REFERENCE):
    lines = ["", "== 3. PAIRED CONTRASTS ===================================================="]
    if reference_label in EXTERNAL_REFERENCES:
        reference = EXTERNAL_REFERENCES[reference_label]
        reference_name = reference_label
        lines.append(
            "  NOTE: this reference comes from a DIFFERENT commit. Between-arm contrasts inside\n"
            "  this wave are the trustworthy ones; this row orients against prior work only.")
    else:
        reference = results.get(reference_label)
        if reference is None:
            lines.append(f"  in-wave reference {reference_label!r} not complete; "
                         f"falling back to external {DEFAULT_REFERENCE}")
            reference = EXTERNAL_REFERENCES[DEFAULT_REFERENCE]
            reference_name = DEFAULT_REFERENCE
        else:
            reference_name = reference_label
    lines.append(f"  reference: {reference_name}")
    lines.append("| Arm | d OOD val | d OOD test | d Worst test | d ID |")
    lines.append("|" + "---|" * 5)
    for label, result in sorted(results.items()):
        if label == reference_name:
            continue
        lines.append(
            f"| {label} | {_delta(result.get('acc_val'), reference.get('acc_val'))} "
            f"| {_delta(result.get('acc_heldout'), reference.get('acc_heldout'))} "
            f"| {_delta(result.get('worst_env_heldout'), reference.get('worst_env_heldout'))} "
            f"| {_delta(result.get('acc_within'), reference.get('acc_within'))} |")
    lines.append("  Deltas are ABSOLUTE PERCENTAGE POINTS. Single seed: these are design signal.")
    return lines


def render_mechanism(results):
    lines = ["", "== 4. MECHANISM GATES ====================================================",
             f"  route_reliance gate: {RELIANCE_GATE} "
             "(completed campaigns never exceeded 0.0065)"]
    for label, result in sorted(results.items()):
        reliance = result.get("route_reliance")
        cosine = result.get("expert_output_cosine")
        verdict = []
        if reliance is None:
            verdict.append("reliance not measurable")
        elif float(reliance) >= RELIANCE_GATE:
            verdict.append("RELIANCE GATE PASSED")
        else:
            verdict.append("below reliance gate")
        if cosine is not None:
            verdict.append("experts near-interchangeable" if float(cosine) > 0.9
                           else "experts differentiated")
        grads = result.get("router_grad_norm_by_epoch") or {}
        if grads:
            epochs = sorted(grads, key=lambda k: int(k))
            first = _first_norm(grads[epochs[0]])
            last = _first_norm(grads[epochs[-1]])
            if first and last is not None:
                ratio = last / first if first else None
                if ratio is not None:
                    verdict.append(
                        f"router grad ep{epochs[0]}->ep{epochs[-1]} x{ratio:.3g}"
                        + (" (VANISHED)" if ratio < 0.05 else ""))
        lines.append(f"  {label}: reliance={_num(reliance)}  expcos={_num(cosine, places=3)}"
                     f"  | {'; '.join(verdict)}")
    lines.append(
        "  A gain with reliance below the gate is extra capacity, not conditional computation.")
    return lines


def _first_norm(mapping):
    if not isinstance(mapping, dict) or not mapping:
        return None
    values = [float(v) for v in mapping.values() if v is not None]
    return values[0] if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="paired reference: an in-wave arm label, or one of the external "
                             "prior-wave references (the default)")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero if any protocol check fails")
    args = parser.parse_args()

    root, manifest, results = load_wave(args.result_root)
    problems = validate(results)

    report = [f"frontier-MoE wave: {len(results)}/{len(manifest['runs'])} complete under {root}"]
    report.append(f"placement fixed at blocks {manifest.get('placement_fixed')}; "
                  f"label smoothing {manifest.get('label_smoothing_fixed')}; "
                  f"mechanism auditing {manifest.get('mechanism_auditing')}")
    if problems:
        report.append("")
        report.append("== PROTOCOL PROBLEMS (resolve before ranking anything) ==")
        report.extend(f"  ! {problem}" for problem in problems)
    report += render_ceilings(results)
    report += render_table(manifest, results)
    report += render_contrasts(results, args.reference)
    report += render_mechanism(results)
    text = "\n".join(report)
    print(text)

    if args.json_out:
        payload = {
            "schema_version": 1,
            "campaign": manifest.get("campaign"),
            "result_root": str(root),
            "complete": len(results),
            "expected": len(manifest["runs"]),
            "selection_split": "ood_val",
            "reliance_gate": RELIANCE_GATE,
            "protocol_problems": problems,
            "arms": {
                label: {
                    **{key: result.get(key) for key in METRICS},
                    "variant": result.get("variant"),
                    "route_reliance": result.get("route_reliance"),
                    "randomized_routes_acc": result.get("randomized_routes_acc"),
                    "shared_only_acc": result.get("shared_only_acc"),
                    "expert_output_cosine": result.get("expert_output_cosine"),
                    "experts_used": result.get("experts_used"),
                    "routing_entropy": result.get("routing_entropy"),
                    "routing_mi_site": result.get("routing_mi_site"),
                    "router_grad_norm_by_epoch": result.get("router_grad_norm_by_epoch"),
                }
                for label, result in sorted(results.items())
            },
            "external_references": EXTERNAL_REFERENCES,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")

    if args.strict and problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
