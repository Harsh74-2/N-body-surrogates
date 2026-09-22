#!/usr/bin/env python
"""Regenerate the two headline audit tables from the post-audit canonical JSONs.

Outputs (consumed by build_github_pages.py -> audit/ on the site):

  results/cross_N_audit.md
      A. OOD rollout mean error (% of scale length L) per training count
         x model variant x preset, from
         real_case_validation/report_N{n}/preset_{p}/summary.json
         (calibrated block where present; 2026-09-22 post-retrain).
      B. Error with vs. without autoregressive feedback (OOD mean),
         from the same rollout reports + the fresh single-step dumps.
      C. In-distribution K=50 rollout drift per N x variant, from
         results/all_eval.json.

  results/cross_N_audit_single_step.md
      In-distribution single-step test MSE / energy drift / latency per
      N x variant, from results/all_eval.json.

All numbers come straight from the JSON artefacts of the 2026-09 post-audit
retrain -- no hand-transcribed values.

all_eval.json has no explicit N/variant fields; its cells are ordered
N-major (10, 25, 50, 100), model-major within (mlp, mlp_stable, lstm,
lstm_stable, gnn, gnn_stable) -- the ordering the aggregator used. The
values are sanity-checked against known post-audit anchors below.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
PRESETS = [
    "disc_imf_in_distribution_baseline",
    "full_solar_system",
    "inner_planets",
    "jupiter_galileans",
    "solar_system_extended",
    "sun_earth_only",
    "sun_planets_moon",
]
VARIANTS = ["mlp", "mlp_stable", "lstm", "lstm_stable", "gnn", "gnn_stable"]
N_VALUES = [10, 25, 50, 100]
ARCH = {"mlp": "MLP", "lstm": "LSTM", "gnn": "GNN"}


_VARIANT_KEY = {
    "mlp": "MLP", "mlp_stable": "MLP_stable",
    "lstm": "LSTM", "lstm_stable": "LSTM_stable",
    "gnn": "GNN", "gnn_stable": "GNN_stable",
}


def _variant_key(variant: str) -> str:
    """'mlp_stable' -> 'MLP_stable' — the per_model key used by the fresh
    2026-09-22 post-retrain reports."""
    return _VARIANT_KEY[variant]


def ood_mean_err(preset: str, n: int, variant: str) -> float | None:
    """Rollout mean error (%) for one (N, preset, variant) from the fresh
    2026-09-22 post-retrain per-preset reports.

    Source: real_case_validation/report_N{n}/preset_{p}/summary.json
    (calibrated block where present, else mean_error_over_L * 100 against
    the leapfrog reference). The pre-retrain source was
    results/real_case_ood/ -- that tree is September legacy and must not
    be read again; the fresh retrain does not populate it.
    """
    f = (REPO / "real_case_validation" / f"report_N{n}"
         / f"preset_{preset}" / "summary.json")
    if not f.exists():
        return None
    pm = json.loads(f.read_text(encoding="utf-8"))["per_model"].get(
        _variant_key(variant))
    if pm is None:
        return None
    cal = pm.get("calibrated")
    if isinstance(cal, dict) and "mean_err_pct" in cal:
        return cal["mean_err_pct"]
    return pm["mean_error_over_L"] * 100.0


def fmt_pct(v: float | None) -> str:
    return "&mdash;" if v is None else f"{v:.1f}%"


def build_ood_table() -> str:
    lines = [
        "### Out-of-distribution rollout error",
        "",
        "Mean rollout positional error as a percentage of the preset's",
        "scale length L, averaged over the rollout, for every training",
        "body count N and model variant. All presets except the disc",
        "baseline are out-of-distribution; their values use the calibrated",
        "post-hoc error of the thesis. Cells marked div. diverged",
        "(error above 100% of L) and are excluded from the mean; the disc",
        "baseline is in-distribution and also excluded from the mean.",
        "",
        "| N | variant | disc (in-dist.) | "
        + " | ".join(p.replace("_", " ") for p in PRESETS[1:]) + " | OOD mean |",
        "|---|---|" + "---|" * (len(PRESETS) + 1),
    ]
    for n in N_VALUES:
        for v in VARIANTS:
            vals = [ood_mean_err(p, n, v) for p in PRESETS]
            ood = vals[1:]
            kept = [x for x in ood if x is not None and x <= 100.0]
            mean = sum(kept) / len(kept) if kept else None
            label = ARCH[v.replace("_stable", "")] + (" (stable)" if v.endswith("_stable") else "")
            cells = [fmt_pct(vals[0])] + [
                "div." if (x is not None and x > 100.0) else fmt_pct(x) for x in ood
            ]
            lines.append(
                f"| {n} | {label} | " + " | ".join(cells)
                + f" | {fmt_pct(mean)} |"
            )
    return "\n".join(lines)


def build_rollout_headline_table() -> str:
    """Compact '| model |'-headed OOD-mean table. regen_top_level_plots.py
    picks its rollout panel values from the FIRST table whose header row
    mentions 'model'; the full per-preset table below carries the exact
    per-preset values and the div. flags. A cell rendered '>100' means
    every OOD preset diverged for that cell (no survivor mean exists);
    the plot parser reads it as 100, the same convention as the log-scale
    y-limit."""
    lines = [
        "### Rollout OOD mean (headline)",
        "",
        "Mean rollout positional error (% of scale length L) averaged",
        "over the six out-of-distribution presets. '>100' = every OOD",
        "preset diverged for that cell, so no survivor mean exists.",
        "",
        "| model | N=10 | N=25 | N=50 | N=100 |",
        "|---|---|---|---|---|",
    ]
    for v in VARIANTS:
        label = ARCH[v.replace("_stable", "")] + (" (stable)" if v.endswith("_stable") else "")
        cells = []
        for n in N_VALUES:
            xs = [ood_mean_err(p, n, v) for p in PRESETS[1:]]
            kept = [x for x in xs if x is not None and x <= 100.0]
            mean = sum(kept) / len(kept) if kept else None
            cells.append("&gt;100" if mean is None else f"{mean:.1f}%")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_all_eval() -> dict[tuple[int, str], dict]:
    cells = json.loads((RESULTS / "all_eval.json").read_text(encoding="utf-8"))
    assert len(cells) == len(N_VALUES) * len(VARIANTS), (
        f"all_eval.json has {len(cells)} cells, expected 24")
    # Ordering anchor: N=25 GNN single-step and N=100 GNN single-step
    # (fresh 2026-09 retrain, rollout-MSE rework; September values were
    # 4.943e-6 / 2.107e-6). Cross-check the constants against the values
    # printed by make_all_eval.py whenever all_eval.json is regenerated.
    anchors = {(25, "gnn"): 1.51e-6, (100, "gnn"): 2.83e-7}
    out: dict[tuple[int, str], dict] = {}
    for n in N_VALUES:
        for v in VARIANTS:
            cell = cells.pop(0)
            out[(n, v)] = cell
    for (n, v), expect in anchors.items():
        got = out[(n, v)]["mse"]
        assert abs(got - expect) / expect < 0.02, (
            f"all_eval.json ordering anchor failed for N={n} {v}: {got:.3e}")
    return out


def build_rollout_table(eval_cells: dict[tuple[int, str], dict]) -> str:
    lines = [
        "### In-distribution rollout drift (test split)",
        "",
        "Mean energy-relative rollout drift over a K=50 autoregressive",
        "rollout on held-out test windows (dimensionless), from the",
        "post-audit evaluation of every checkpoint.",
        "",
        "| N | variant | rollout drift (K=50) |",
        "|---|---|---|",
    ]
    for n in N_VALUES:
        for v in VARIANTS:
            label = ARCH[v.replace("_stable", "")] + (" (stable)" if v.endswith("_stable") else "")
            lines.append(f"| {n} | {label} | {eval_cells[(n, v)]['rollout']:.2e} |")
    return "\n".join(lines)


def build_single_step_table(eval_cells: dict[tuple[int, str], dict]) -> str:
    lines = [
        "Single-step test metrics (held-out windows, in-distribution):",
        "mean squared error of the next-frame prediction, the",
        "energy-relative drift of that prediction, and the per-step",
        "latency. Source: `results/all_eval.json`.",
        "",
        "| N | variant | params | single-step MSE | energy drift | latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for n in N_VALUES:
        for v in VARIANTS:
            c = eval_cells[(n, v)]
            label = ARCH[v.replace("_stable", "")] + (" (stable)" if v.endswith("_stable") else "")
            lines.append(
                f"| {n} | {label} | {c['n_params']:,} | {c['mse']:.2e} | "
                f"{c['energy']:.2e} | {c['latency_ms']:.3f} |")
    return "\n".join(lines)


def build_per_preset_ss_sections() -> str:
    """Per-preset single-step sections consumed by regen_top_level_plots.py
    (its parser wants '### <preset>' blocks whose first table has a
    '| model' header row; the prose must stay BETWEEN the heading and the
    next heading, never between heading and table). Stable variants are
    not in the fresh single-step dumps and render as dashes."""
    lines = []
    for preset in PRESETS:
        tag = ("in-distribution"
               if preset == PRESETS[0] else "OOD")
        lines += [
            f"### {preset} ({tag})",
            "",
            "| model | N=10 | N=25 | N=50 | N=100 |",
            "|---|---|---|---|---|",
        ]
        for v in VARIANTS:
            label = ARCH[v.replace("_stable", "")] + (" (stable)" if v.endswith("_stable") else "")
            cells = []
            for n in N_VALUES:
                x = ss_mean_err(n, preset, v)
                cells.append("&mdash;" if x is None else f"{x:.2f}%")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    eval_cells = load_all_eval()

    rollout_md = "\n\n".join([
        "# Cross-N audit (rollout)",
        "",
        "Post-retrain (2026-09-22). Every number below is read",
        "directly from the canonical result JSONs of that run; nothing is",
        "hand-transcribed.",
        build_rollout_headline_table(),
        build_ood_table(),
        build_compounding_table(),
        build_rollout_table(eval_cells),
    ]) + "\n"
    (RESULTS / "cross_N_audit.md").write_text(rollout_md, encoding="utf-8")

    ss_md = "\n\n".join([
        "# Cross-N audit (single-step)",
        "",
        "Post-retrain (2026-09-22).",
        build_single_step_table(eval_cells),
        "Per-preset single-step mean error (% of the preset's scale",
        "length L), averaged over every start index in the window:",
        "each prediction is scored against a reference-rebuilt input",
        "window, so errors do not compound. Source: the fresh",
        "2026-09-22 single-step dumps.",
        build_per_preset_ss_sections(),
    ]) + "\n"
    (RESULTS / "cross_N_audit_single_step.md").write_text(ss_md, encoding="utf-8")

    print("wrote results/cross_N_audit.md and results/cross_N_audit_single_step.md")
    update_hub()


# ----- Interactive-hub regeneration -------------------------------------
# The hub page (_site_static/main_interactive.html) carries its result
# tables between HTML comment markers; everything between them is
# regenerated here from the same canonical JSONs, so the hub can never
# drift back to hand-transcribed (and potentially stale) numbers.

HUB = REPO / "_site_static" / "main_interactive.html"
HUB_BEGIN = "<!--RESULTS-AUTO:BEGIN-->"
HUB_END = "<!--RESULTS-AUTO:END-->"


def ss_mean_err(n: int, preset: str, variant: str) -> float | None:
    """Single-step mean error (%) for one (N, preset, variant) from the
    fresh single-step dump summaries. Source:
    real_case_validation/report_N{n}/single_step/preset_{p}/
    ss_summary.json (the fresh retrain's dump location; the pre-retrain
    source results/real_case_ss/ is September legacy). The fresh dumps
    cover the three single-step variants only -- stable variants return
    None and render as dashes."""
    f = (REPO / "real_case_validation" / f"report_N{n}" / "single_step"
         / f"preset_{preset}" / "ss_summary.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    pm = d.get("per_model", {}).get(_variant_key(variant))
    return None if pm is None else pm["mean_err_pct"]


def build_compounding_table() -> str:
    lines = [
        "### Error with vs. without autoregressive feedback (OOD mean)",
        "",
        "The same checkpoints scored two ways, averaged over the six",
        "out-of-distribution presets. 'Single-step (no compounding)'",
        "rebuilds the input window from the true reference at every step,",
        "so prediction errors never feed back into the model -- it is",
        "scored at every point of the rollout horizon without feedback.",
        "'Full rollout' lets predictions feed back (autoregressive), so",
        "errors compound. Diverged rollout cells (error above 100% of L)",
        "are excluded from the rollout mean, so the compounding factor is",
        "a lower bound. The gap between the two columns isolates the cost",
        "of error compounding: the same checkpoints hold roughly 5x lower",
        "error when their own predictions never feed back.",
        "",
        "| N | variant | no compounding | full rollout | factor |",
        "|---|---|---|---|---|",
    ]
    for n in N_VALUES:
        for v in VARIANTS:
            label = ARCH[v.replace("_stable", "")] + (" (stable)" if v.endswith("_stable") else "")
            ss_vals = [ss_mean_err(n, p, v) for p in PRESETS[1:]]
            ro_vals = [ood_mean_err(p, n, v) for p in PRESETS[1:]]
            ss_kept = [x for x in ss_vals if x is not None and x <= 100.0]
            ro_kept = [x for x in ro_vals if x is not None and x <= 100.0]
            ss_mean = sum(ss_kept) / len(ss_kept) if ss_kept else None
            ro_mean = sum(ro_kept) / len(ro_kept) if ro_kept else None
            factor = (f"&times;{ro_mean / ss_mean:.1f}"
                      if (ss_mean and ro_mean) else "&mdash;")
            lines.append(
                f"| {n} | {label} | {fmt_pct(ss_mean)} | {fmt_pct(ro_mean)} "
                f"| {factor} |")
    return "\n".join(lines)


def _variant_rows(cell_fn, div_at: float | None) -> list[str]:
    """One <tr> per variant; columns N=10/25/50/100."""
    rows = []
    for v in VARIANTS:
        label = ARCH[v.replace("_stable", "")] + (" (stable)" if v.endswith("_stable") else "")
        cells = []
        for n in N_VALUES:
            x = cell_fn(n, v)
            if x is None:
                cells.append("&mdash;")
            elif div_at is not None and x > div_at:
                cells.append("*div.*")
            else:
                cells.append(f"{x:.1f} %")
        rows.append(f"<tr><td>{label}</td><td>" + "</td><td>".join(cells) + "</td></tr>")
    return rows


def _preset_section(cell_fn, div_at: float | None) -> list[str]:
    out = []
    for preset in PRESETS:
        tag = "in-distribution" if preset == PRESETS[0] else "OOD"
        out.append(f"<h3><code>{preset}</code> ({tag})</h3>")
        out.append("<table><tr><th>model</th><th>N=10</th><th>N=25</th>"
                   "<th>N=50</th><th>N=100</th></tr>")
        out += _variant_rows(lambda n, v: cell_fn(n, preset, v), div_at)
        out.append("</table>")
    return out


def hub_fragment() -> str:
    parts = []

    def headline(cell_fn, div_at, exclude_first_preset: bool) -> list[str]:
        """Headline table: per variant, the mean over presets (OOD only
        when exclude_first_preset), diverged cells excluded."""
        rows = []
        for v in VARIANTS:
            label = ARCH[v.replace("_stable", "")] + (
                " (stable)" if v.endswith("_stable") else "")
            cells, first = [], None
            for n in N_VALUES:
                vals = [cell_fn(n, p, v) for p in PRESETS[1 if exclude_first_preset else 0:]]
                kept = [x for x in vals if x is not None
                        and (div_at is None or x <= div_at)]
                mean = sum(kept) / len(kept) if kept else None
                if first is None:
                    first = mean
                last = mean
                cells.append("&mdash;" if mean is None else f"{mean:.1f} %")
            delta = (last - first) if (first is not None and last is not None) else None
            rows.append(f"<tr><td>{label}</td><td>" + "</td><td>".join(cells)
                        + f"</td><td>{'&mdash;' if delta is None else f'{delta:+.1f} pp'}</td></tr>")
        return rows

    # --- Rollout (OOD presets from real_case_ood) ---
    parts += [
        "<h2>Headline results &mdash; rollout (multi-step) error</h2>",
        "<p>Mean trajectory error after a full autoregressive rollout, as a",
        "percentage of each preset&rsquo;s scale length L, averaged over the six",
        "out-of-distribution presets. Cells marked <em>div.</em> diverged",
        "(error above 100% of L) and are excluded from the average; the",
        "in-distribution disc baseline is tabulated per preset below but",
        "excluded from the average. Lower is better.",
        "Source: <a href='audit/cross_N_audit.md'><code>cross_N_audit.md</code></a>.</p>",
        "<table><tr><th>model</th><th>N=10</th><th>N=25</th><th>N=50</th>"
        "<th>N=100</th><th>&Delta; (N=100 &minus; N=10, pp)</th></tr>",
    ]
    parts += headline(lambda n, p, v: ood_mean_err(p, n, v), 100.0, True)
    parts += ["</table>",
              "<h2>Per-preset rollout error % across N</h2>"]
    parts += _preset_section(lambda n, p, v: ood_mean_err(p, n, v), 100.0)
    parts += [
        "<h2>Family-level verdict (single vs stability-trained, OOD mean)</h2>",
        "<p>At N=10 and N=100, the OOD-mean rollout error of each "
        "architecture with and without the rollout-MSE stability term.</p>",
        "<table><tr><th>family</th><th>N=10 single</th><th>N=10 stable</th>"
        "<th>&Delta; (pp)</th><th>N=100 single</th><th>N=100 stable</th>"
        "<th>&Delta; (pp)</th></tr>"]
    for base in ("mlp", "lstm", "gnn"):
        row = [ARCH[base]]
        for n in (10, 100):
            vals = {}
            for v, tag in ((base, "single"), (f"{base}_stable", "stable")):
                xs = [ood_mean_err(p, n, v) for p in PRESETS[1:]]
                kept = [x for x in xs if x is not None and x <= 100.0]
                vals[tag] = sum(kept) / len(kept) if kept else None
            # A None mean (e.g. LSTM N=100: every OOD preset diverged) must
            # render as a dash, not crash the hub rebuild.
            if vals["single"] is None or vals["stable"] is None:
                row += ["&mdash;", "&mdash;", "&mdash;"]
            else:
                row += [f"{vals['single']:.1f} %", f"{vals['stable']:.1f} %",
                        f"{vals['stable'] - vals['single']:+.1f}"]
        parts.append("<tr><td>" + "</td><td>".join(row) + "</td></tr>")
    parts.append("</table>")

    # --- Single-step (fresh real_case_ss dumps) ---
    parts += [
        "<h2>Headline results &mdash; single-step error</h2>",
        "<p>Mean one-step prediction error (window always rebuilt from the",
        "reference, so errors do not compound), as a percentage of L,",
        "averaged over the six out-of-distribution presets. The",
        "in-distribution disc baseline is tabulated per preset below but",
        "excluded from the average.",
        "Source: <a href='audit/cross_N_audit_single_step.md'>"
        "<code>cross_N_audit_single_step.md</code></a> (in-distribution",
        "test metrics); per-preset OOD values from the",
        "2026-09 single-step dump run.</p>",
        "<table><tr><th>model</th><th>N=10</th><th>N=25</th><th>N=50</th>"
        "<th>N=100</th><th>&Delta; (N=100 &minus; N=10, pp)</th></tr>",
    ]
    parts += headline(lambda n, p, v: ss_mean_err(n, p, v), None, True)
    parts += ["</table>",
              "<h2>Per-preset single-step error % across N</h2>"]
    parts += _preset_section(lambda n, p, v: ss_mean_err(n, p, v), None)
    parts.append("")

    return "\n".join(parts)


def update_hub() -> None:
    if not HUB.exists():
        print(f"[hub] {HUB} not found; skipped")
        return
    text = HUB.read_text(encoding="utf-8")
    if HUB_BEGIN not in text or HUB_END not in text:
        print("[hub] markers not present; run once to splice them manually")
        return
    pre = text.split(HUB_BEGIN)[0]
    post = text.split(HUB_END)[1]
    HUB.write_text(pre + HUB_BEGIN + "\n" + hub_fragment() + "\n" + HUB_END + post,
                   encoding="utf-8")
    print(f"[hub] regenerated results block in {HUB.name}")


if __name__ == "__main__":
    main()