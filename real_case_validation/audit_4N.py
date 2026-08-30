"""
audit_4N.py
===========
Side-by-side error-percentage audit across the N = 10, 25, 50, 100
training-budget reruns. Reads every `report_N<NN>/preset_*/summary.json`
(or `report_N<NN>/single_step/preset_*/ss_summary.json` for the
single-step variant) and produces a single Markdown comparison that
shows the gradual improvement (or absence thereof) of surrogate
accuracy as the training budget N grows.

Usage
-----
    python -m real_case_validation.audit_4N \
        --reports-dir real_case_validation \
        --out        real_case_validation/cross_N_audit.md

    # Single-step variant:
    python -m real_case_validation.audit_4N \
        --reports-dir real_case_validation \
        --single-step \
        --out        real_case_validation/cross_N_audit_single_step.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Stable ordering so the report is deterministic across reruns.
MODEL_ORDER = ("MLP", "MLP_stable", "LSTM", "LSTM_stable",
               "GNN", "GNN_stable")
N_VALUES = (10, 25, 50, 100)


def _load_one(report_dir: Path, n: int, single_step: bool) -> dict:
    """Per-N: load every preset summary, return dict[preset_name] -> payload."""
    sub = report_dir / f"report_N{n}"
    if not sub.exists():
        return {}
    if single_step:
        sub = sub / "single_step"
        if not sub.exists():
            return {}
    out = {}
    for preset_dir in sorted(sub.glob("preset_*")):
        if single_step:
            f = preset_dir / "ss_summary.json"
        else:
            f = preset_dir / "summary.json"
        if not f.exists():
            continue
        with open(f, "r", encoding="utf-8") as fh:
            out[preset_dir.name[len("preset_"):]] = json.load(fh)
    return out


def _row(payload: dict, model: str, single_step: bool) -> dict | None:
    pm = payload.get("per_model", {}).get(model)
    if pm is None:
        return None
    if single_step:
        return {
            "model":         model,
            "mean_err_pct":  pm["mean_err_pct"],
            "max_err_pct":   pm["max_err_pct"],
            "mse_position":  pm["mse_position"],
        }
    return {
        "model":         model,
        "mean_err_pct":  100.0 * pm["mean_error_over_L"],
        "max_err_pct":   100.0 * pm["max_error_over_L"],
        "mse_position":  pm["mse_position"],
    }


def _aggregate_per_model(presets: dict, single_step: bool) -> dict:
    """Mean of mean_err_% across the presets in this N bucket, per model."""
    out = {m: [] for m in MODEL_ORDER}
    for p in presets.values():
        for m in MODEL_ORDER:
            r = _row(p, m, single_step)
            if r is not None:
                out[m].append(r["mean_err_pct"])
    return out


def _render_md(per_n: dict[int, dict], single_step: bool) -> str:
    lines = []
    lines.append("# Cross-N Real-Case Validation, Error-Percentage Audit\n")
    if single_step:
        lines.append(
            "Single-step variant: each surrogate predicts the next "
            "frame only from a warm-up window of leapfrog frames. "
            "Errors do **not** compound because the window is always "
            "re-built from the reference. This is the headline 1-3 % "
            "single-step MSE the surrogates were trained on.\n")
    else:
        lines.append(
            "Autoregressive rollout variant: each surrogate predicts "
            "**forward in time** from its own previous output. Errors "
            "compound over the rollout. Use this report to read the "
            "distribution-shift cost of the Solar System relative to "
            "the synthetic disc training set.\n")
    lines.append(
        "Each cell is the *mean error %* averaged across every preset "
        "that ran for that N. The cell on the right (N=100) is the "
        "**best** any model in this family can do on the given "
        "training budget; the cell on the left (N=10) is the worst. "
        "Reading left-to-right should show the gradual improvement "
        "as the training budget grows.\n")

    # ── Per-N, per-model table (the headline) ─────────────────────────
    lines.append("## Headline: mean error % by (N, model)\n")
    lines.append("| model | N=10 | N=25 | N=50 | N=100 | "
                 "Δ (N=100 − N=10, pp) |")
    lines.append("|---|---|---|---|---|---|")
    for m in MODEL_ORDER:
        cells = []
        deltas = []
        for n in N_VALUES:
            presets = per_n.get(n, {})
            agg = _aggregate_per_model(presets, single_step)
            if agg[m]:
                cells.append(f"{sum(agg[m]) / len(agg[m]):.1f} %")
            else:
                cells.append("—")
        # Δ N=100 − N=10
        agg10 = _aggregate_per_model(per_n.get(10, {}), single_step)
        agg100 = _aggregate_per_model(per_n.get(100, {}), single_step)
        if agg10[m] and agg100[m]:
            delta = (sum(agg100[m]) / len(agg100[m])
                     - sum(agg10[m]) / len(agg10[m]))
            deltas.append(f"{delta:+.1f}")
        else:
            deltas.append("—")
        lines.append(f"| {m} | {cells[0]} | {cells[1]} | {cells[2]} | "
                     f"{cells[3]} | {deltas[0]} |")
    lines.append("")

    # ── Per-preset detail (one row per preset, all 4 N values) ────────
    lines.append("## Per-preset mean error % across N (headline preset detail)\n")
    # Discover every preset name across all 4 N values.
    all_presets = set()
    for presets in per_n.values():
        all_presets.update(presets.keys())
    for preset_name in sorted(all_presets):
        in_dist = any(per_n[n].get(preset_name, {}).get("in_distribution", False)
                      for n in N_VALUES)
        tag = "in-distribution" if in_dist else "OOD"
        lines.append(f"### `{preset_name}` ({tag})\n")
        lines.append("| model | N=10 | N=25 | N=50 | N=100 |")
        lines.append("|---|---|---|---|---|")
        for m in MODEL_ORDER:
            cells = []
            for n in N_VALUES:
                payload = per_n.get(n, {}).get(preset_name)
                if payload is None:
                    cells.append("—")
                    continue
                r = _row(payload, m, single_step)
                if r is None:
                    cells.append("—")
                else:
                    cells.append(f"{r['mean_err_pct']:.2f} %")
            lines.append(f"| {m} | {cells[0]} | {cells[1]} | "
                         f"{cells[2]} | {cells[3]} |")
        lines.append("")

    # ── Family-level verdict ───────────────────────────────────────────
    lines.append("## Family-level verdict (single vs stable, mean across N)\n")
    lines.append("| family | N=10 single | N=10 stable | Δ (pp) | "
                 "N=100 single | N=100 stable | Δ (pp) |")
    lines.append("|---|---|---|---|---|---|---|")
    for base in ("MLP", "LSTM", "GNN"):
        agg10 = _aggregate_per_model(per_n.get(10, {}), single_step)
        agg100 = _aggregate_per_model(per_n.get(100, {}), single_step)
        s10 = sum(agg10[base]) / len(agg10[base]) if agg10[base] else float("nan")
        st10 = sum(agg10[f"{base}_stable"]) / len(agg10[f"{base}_stable"]) \
            if agg10[f"{base}_stable"] else float("nan")
        s100 = sum(agg100[base]) / len(agg100[base]) if agg100[base] else float("nan")
        st100 = sum(agg100[f"{base}_stable"]) / len(agg100[f"{base}_stable"]) \
            if agg100[f"{base}_stable"] else float("nan")
        lines.append(
            f"| {base} | {s10:.2f} % | {st10:.2f} % | {st10 - s10:+.2f} | "
            f"{s100:.2f} % | {st100:.2f} % | {st100 - s100:+.2f} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", required=True,
                    help="Parent directory containing report_N<NN>/ "
                         "subdirs.")
    ap.add_argument("--out", required=True,
                    help="Path to write the Markdown audit.")
    ap.add_argument("--single-step", action="store_true",
                    help="Audit the single-step variant instead of "
                         "the autoregressive rollout.")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir).expanduser().resolve()
    if not reports_dir.exists():
        raise SystemExit(f"reports dir not found: {reports_dir}")
    per_n: dict[int, dict] = {}
    for n in N_VALUES:
        per_n[n] = _load_one(reports_dir, n, single_step=args.single_step)
        if not per_n[n]:
            print(f"[warn] no data for N={n} (report_N{n}/"
                  f"{'single_step/' if args.single_step else ''} not found)")
    md = _render_md(per_n, single_step=args.single_step)
    out_md = Path(args.out).expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[md] {out_md}")


if __name__ == "__main__":
    main()
