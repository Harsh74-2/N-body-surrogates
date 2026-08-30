"""
audit_single_N.py
=================
Per-N error-percentage audit. Reads a single `report_N<NN>/` directory
and produces a self-contained Markdown report with:
  - Headline: mean err % per model (single + stable, all 6 variants)
  - Per-preset detail tables
  - Stable-vs-single family verdict
  - Single-step + rollout side-by-side comparison

Usage
-----
    python -m real_case_validation.audit_single_N \
        --report-dir real_case_validation/report_N50 \
        --out        real_case_validation/report_N50/N50_audit.md

The output is meant to stand alone — every chart, table, and verdict
needed for the "N=NN real-life validation" subsection is here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ORDER = ("MLP", "MLP_stable", "LSTM", "LSTM_stable",
               "GNN", "GNN_stable")


def _load_one(report_dir: Path, single_step: bool) -> dict:
    """Per-preset summary.json (or ss_summary.json) keyed by preset name."""
    sub = report_dir / ("single_step" if single_step else "")
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
            "mse_state":     pm["mse_state"],
            "energy_drift":  pm["energy_drift"],
        }
    return {
        "model":         model,
        "mean_err_pct":  100.0 * pm["mean_error_over_L"],
        "max_err_pct":   100.0 * pm["max_error_over_L"],
        "mse_position":  pm["mse_position"],
        "mse_state":     pm["mse_state"],
        "frames_before_half_L": pm["frames_before_half_L"],
        "energy_drift":  pm["max_energy_drift"],
    }


def _aggregate(presets: dict, single_step: bool) -> dict:
    """Mean err % per model across all presets in this N bucket."""
    out = {m: [] for m in MODEL_ORDER}
    for p in presets.values():
        for m in MODEL_ORDER:
            r = _row(p, m, single_step)
            if r is not None:
                out[m].append(r["mean_err_pct"])
    return out


def _render_md(report_dir: Path,
               rollout: dict, ss: dict | None,
               n: int) -> str:
    lines = []
    lines.append(f"# Real-Case Validation, N = {n}\n")
    lines.append(
        f"Standalone audit for the **N = {n}** training-budget "
        f"rerun. Two complementary modes are reported:\n")
    lines.append(
        "- **Autoregressive rollout** (in `preset_*/summary.json`) — "
        "each surrogate predicts forward from its own previous "
        "output. Errors compound; the per-step prediction becomes "
        "the *warm-up window* for the next. This is the **stress "
        "test** the user cares about: *how far can the model "
        "extrapolate before it loses the orbit?*")
    lines.append(
        "- **Single-step variant** (in `preset_*/ss_summary.json`) — "
        "each surrogate predicts the next frame *only*, with the "
        "warm-up window always re-built from the leapfrog reference "
        "(never from the model's own output). Errors do not "
        "compound. This is the **bare prediction error** and the "
        "headline 1-3 % number the surrogates were trained on.\n")

    in_dist = [p for p in rollout.values() if p.get("in_distribution")]
    ood     = [p for p in rollout.values() if not p.get("in_distribution")]

    # ── Headline: rollout (autoregressive) ────────────────────────────
    if rollout:
        lines.append(f"## Headline (autoregressive rollout, mean err %)\n")
        agg_all = _aggregate(rollout, single_step=False)
        agg_in  = _aggregate({k: v for k, v in rollout.items() if v.get("in_distribution")}, single_step=False)
        agg_ood = _aggregate({k: v for k, v in rollout.items() if not v.get("in_distribution")}, single_step=False)
        lines.append("Each cell is the mean of `mean_err_%` over the "
                     "presets that ran, normalised by L.\n")
        lines.append("| model | in-distribution | Solar-System OOD | all |")
        lines.append("|---|---|---|---|")
        for m in MODEL_ORDER:
            in_avg = (sum(agg_in[m]) / len(agg_in[m])
                      if agg_in[m] else float("nan"))
            od_avg = (sum(agg_ood[m]) / len(agg_ood[m])
                      if agg_ood[m] else float("nan"))
            all_avg = (sum(agg_all[m]) / len(agg_all[m])
                       if agg_all[m] else float("nan"))
            lines.append(
                f"| {m} | {in_avg:.1f} % | {od_avg:.1f} % | {all_avg:.1f} % |")
        lines.append("")

    # ── Headline: single-step ─────────────────────────────────────────
    if ss:
        lines.append("## Headline (single-step, mean err %)\n")
        agg_all = _aggregate(ss, single_step=True)
        agg_in  = _aggregate({k: v for k, v in ss.items() if v.get("in_distribution")}, single_step=True)
        agg_ood = _aggregate({k: v for k, v in ss.items() if not v.get("in_distribution")}, single_step=True)
        lines.append("Each cell is the mean of `mean_err_%` over the "
                     "presets that ran. The in-distribution row should "
                     "sit at 1-3 % — this is the **bare** prediction "
                     "error the surrogates were trained on.\n")
        lines.append("| model | in-distribution | Solar-System OOD | all |")
        lines.append("|---|---|---|---|")
        for m in MODEL_ORDER:
            in_avg = (sum(agg_in[m]) / len(agg_in[m])
                      if agg_in[m] else float("nan"))
            od_avg = (sum(agg_ood[m]) / len(agg_ood[m])
                      if agg_ood[m] else float("nan"))
            all_avg = (sum(agg_all[m]) / len(agg_all[m])
                       if agg_all[m] else float("nan"))
            lines.append(
                f"| {m} | {in_avg:.1f} % | {od_avg:.1f} % | {all_avg:.1f} % |")
        lines.append("")

    # ── Stable-vs-single family verdict (per N) ──────────────────────
    if rollout:
        lines.append("## Stable vs single, per family (rollout)\n")
        lines.append("| family | single mean err % | stable mean err % | Δ (pp) |")
        lines.append("|---|---|---|---|")
        agg_ood = _aggregate({k: v for k, v in rollout.items()
                              if not v.get("in_distribution")}, single_step=False)
        for base in ("MLP", "LSTM", "GNN"):
            single = agg_ood.get(base, [])
            stable = agg_ood.get(f"{base}_stable", [])
            if single and stable:
                s_avg = sum(single) / len(single)
                st_avg = sum(stable) / len(stable)
                lines.append(
                    f"| {base} | {s_avg:.2f} % | {st_avg:.2f} % | "
                    f"{st_avg - s_avg:+.2f} |")
        lines.append("")

    if ss:
        lines.append("## Stable vs single, per family (single-step)\n")
        lines.append("| family | single mean err % | stable mean err % | Δ (pp) |")
        lines.append("|---|---|---|---|")
        agg_ood = _aggregate({k: v for k, v in ss.items()
                              if not v.get("in_distribution")}, single_step=True)
        for base in ("MLP", "LSTM", "GNN"):
            single = agg_ood.get(base, [])
            stable = agg_ood.get(f"{base}_stable", [])
            if single and stable:
                s_avg = sum(single) / len(single)
                st_avg = sum(stable) / len(stable)
                lines.append(
                    f"| {base} | {s_avg:.2f} % | {st_avg:.2f} % | "
                    f"{st_avg - s_avg:+.2f} |")
        lines.append("")

    # ── Per-preset detail (rollout) ───────────────────────────────────
    if rollout:
        lines.append("## Per-preset detail (autoregressive rollout)\n")
        for preset_name, payload in sorted(rollout.items()):
            tag = "in-distribution" if payload.get("in_distribution") else "OOD"
            lines.append(f"### `{preset_name}` — {payload.get('label', '')} ({tag})\n")
            lines.append("| model | mean err % | max err % | frames ≤ ½L | "
                         "energy drift | MSE pos |")
            lines.append("|---|---|---|---|---|---|")
            for m in MODEL_ORDER:
                r = _row(payload, m, single_step=False)
                if r is None:
                    continue
                lines.append(
                    f"| {r['model']} | {r['mean_err_pct']:.2f} % | "
                    f"{r['max_err_pct']:.2f} % | "
                    f"{r['frames_before_half_L']} | "
                    f"{r['energy_drift']:.2e} | "
                    f"{r['mse_position']:.3e} |")
            lines.append("")

    # ── Per-preset detail (single-step) ───────────────────────────────
    if ss:
        lines.append("## Per-preset detail (single-step)\n")
        for preset_name, payload in sorted(ss.items()):
            tag = "in-distribution" if payload.get("in_distribution") else "OOD"
            lines.append(f"### `{preset_name}` — {payload.get('label', '')} ({tag})\n")
            lines.append("| model | mean err % | max err % | "
                         "energy drift | MSE pos |")
            lines.append("|---|---|---|---|---|")
            for m in MODEL_ORDER:
                r = _row(payload, m, single_step=True)
                if r is None:
                    continue
                lines.append(
                    f"| {r['model']} | {r['mean_err_pct']:.2f} % | "
                    f"{r['max_err_pct']:.2f} % | "
                    f"{r['energy_drift']:.2e} | "
                    f"{r['mse_position']:.3e} |")
            lines.append("")

    # ── Per-N takeaway ───────────────────────────────────────────────
    lines.append(f"## N = {n} takeaway\n")
    if rollout and ss:
        ss_in = _aggregate({k: v for k, v in ss.items()
                            if v.get("in_distribution")}, single_step=True)
        lines.append(
            f"- Single-step in-distribution baseline (the headline "
            f"1-3 % number): "
            + ", ".join(
                f"{m} = {sum(ss_in[m]) / len(ss_in[m]):.2f} %"
                for m in MODEL_ORDER if ss_in[m])
            + ".")
        if ood:
            lines.append(
                f"- Rollout OOD stability (mean err % over {len(ood)} "
                f"OOD presets): see the headline table above. The "
                f"stable variant of each family is listed explicitly; "
                f"compare to its single-step neighbour to read off "
                f"the training-stability benefit at this N.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", required=True,
                    help="Path to a real_case_validation/report_N<NN>/ "
                         "directory containing preset_*/summary.json "
                         "and (optionally) preset_*/ss_summary.json "
                         "under single_step/.")
    ap.add_argument("--out", required=True,
                    help="Path to write the Markdown audit.")
    ap.add_argument("--n", type=int, default=None,
                    help="N value (used in the title). If omitted, "
                         "inferred from the directory name "
                         "(report_N<NN>/ -> NN).")
    args = ap.parse_args()

    report_dir = Path(args.report_dir).expanduser().resolve()
    if not report_dir.exists():
        raise SystemExit(f"report dir not found: {report_dir}")
    if args.n is not None:
        n = args.n
    else:
        # infer from report_dir.name = "report_N<NN>"
        name = report_dir.name
        if name.startswith("report_N"):
            try:
                n = int(name[len("report_N"):])
            except ValueError:
                n = -1
        else:
            n = -1

    rollout = _load_one(report_dir, single_step=False)
    ss = _load_one(report_dir, single_step=True) \
        if (report_dir / "single_step").exists() else None
    if not rollout and not ss:
        raise SystemExit(f"no summaries found under {report_dir}")

    md = _render_md(report_dir, rollout, ss, n=n)
    out_md = Path(args.out).expanduser().resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[md] {out_md}  (rollout={len(rollout)}, "
          f"single-step={len(ss) if ss else 0} presets)")


if __name__ == "__main__":
    main()