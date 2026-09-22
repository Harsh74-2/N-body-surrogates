#!/usr/bin/env python
"""Regenerate the two headline audit tables from the post-audit canonical JSONs.

Outputs (consumed by build_github_pages.py -> audit/ on the site):

  results/cross_N_audit.md
      A. OOD rollout mean error (% of scale length L) per training count
         x model variant x preset, from results/real_case_ood/*/
         real_case_report.json (calibrated block where present).
      B. In-distribution K=50 rollout drift per N x variant, from
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


def ood_mean_err(preset: str, n: int, variant: str) -> float | None:
    d = RESULTS / "real_case_ood" / f"{preset}_N{n}_{variant}"
    f = d / "real_case_report.json"
    if not f.exists():
        return None
    report = json.loads(f.read_text(encoding="utf-8"))
    entry = report[0]
    arch = ARCH[variant.replace("_stable", "")]
    pm = entry["per_model"].get(arch)
    if pm is None:
        return None
    cal = pm.get("calibrated")
    if cal is not None:
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


def main() -> None:
    eval_cells = load_all_eval()

    rollout_md = "\n\n".join([
        "# Cross-N audit (rollout)",
        "",
        "Post-audit retrain (2026-09). Every number below is read",
        "directly from the canonical result JSONs of that run; nothing is",
        "hand-transcribed.",
        build_ood_table(),
        build_rollout_table(eval_cells),
    ]) + "\n"
    (RESULTS / "cross_N_audit.md").write_text(rollout_md, encoding="utf-8")

    ss_md = "\n\n".join([
        "# Cross-N audit (single-step)",
        "",
        "Post-audit retrain (2026-09).",
        build_single_step_table(eval_cells),
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
    fresh single-step dump summaries."""
    f = (REPO / "results" / "real_case_ss" / f"N{n}" / "single_step"
         / f"preset_{preset}" / "ss_summary.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    pm = d.get("per_model", {}).get(variant)
    return None if pm is None else pm["mean_err_pct"]


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
        "architecture with and without the rollout-energy stability term.</p>",
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