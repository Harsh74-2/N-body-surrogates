#!/usr/bin/env python3
"""
regen_top_level_plots.py
========================
Re-renders the three top-level benchmark plots from the *canonical* data on
disk (`real_case_validation/cross_N_audit_single_step.md` and
`results/latency_bench.json`). The legacy `results/N{n}/metrics.json`
and `results/N{n}/stability.json` files were deleted in the audit pass;
this script does not depend on them.

Outputs:
  plots/eval_benchmark.png    — 4-panel bar chart per N (single-step
                                baseline + OOD mean err % + latency +
                                stability) for MLP/LSTM/GNN.
  plots/stability_N10.png,
  plots/stability_N25.png,
  plots/stability_N50.png,
  plots/stability_N100.png,
  plots/stability_overview.png — rollout-stability gradients across
                                models and N (single-step variant vs
                                stable variant; using OOD mean err %
                                drop over the rollout horizon).
  plots/scaling_latency.png    — per-N latency scaling (solver vs
                                surrogate single-frame, vs surrogate
                                batched amortised), from
                                results/latency_bench.json.

No GPU, no checkpoints — just JSON + markdown parsing + matplotlib.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Theme (matches validation.py so figures sit together) ───────────
THEME = {
    "bg":       "#11151c",
    "panel":    "#161c26",
    "grid":     "#3a4658",
    "text":     "#e7ecf2",
    "spine":    "#7d8597",
    "violet":   "#9b6dff",
    "good":     "#3ddc97",
    "accent":   "#56b6f2",
}
MODEL_COLORS = {
    "MLP":         THEME["violet"],
    "MLP_stable":  "#c79bff",
    "LSTM":        THEME["good"],
    "LSTM_stable": "#9af3c8",
    "GNN":         THEME["accent"],
    "GNN_stable":  "#9bd5ff",
}
MODEL_SHORT = {
    "MLP": "MLP",
    "LSTM": "LSTM",
    "GNN": "GNN",
}


# ── Markdown table parser ────────────────────────────────────────────
def _parse_pipe_table(md_text: str, heading_prefix: str) -> dict[str, list[float]]:
    """Find the first markdown pipe table whose row-0 cell 0 begins
    with `heading_prefix`. Return {row_label: [float, ...]} for every
    body row (assumes the body rows have | N=NN | values | ...)."""
    lines = md_text.splitlines()
    rows = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and heading_prefix in line:
            header_cells = [c.strip() for c in line.strip("|").split("|")]
            # next line is the separator
            if i + 1 >= len(lines):
                return {}
            i += 2  # skip header + separator
            # consume body rows
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                # row label is cell 0
                label = cells[0]
                # remaining cells are N=10 / N=25 / N=50 / N=100 (or
                # n=10 if lowercased)
                vals = []
                for c in cells[1:]:
                    m = re.search(r"([-+]?\d+(?:\.\d+)?)", c)
                    if m:
                        vals.append(float(m.group(1)))
                rows[label] = vals
                i += 1
            return rows
        i += 1
    return rows


def _load_single_step_ood_mean(cross_n_audit_md: Path) -> dict[str, list[float]]:
    """Mean of OOD mean_err_% across the OOD presets, for each
    (model, N), from cross_N_audit_single_step.md. We average across
    the 6 OOD presets (everything except
    `disc_imf_in_distribution_baseline`)."""
    md = cross_n_audit_md.read_text(encoding="utf-8")
    sections: dict[str, dict[str, list[float]]] = {}
    for block in md.split("### ")[1:]:
        head, *rest = block.split("\n", 1)
        preset = head.split(" (")[0].strip().strip("`")
        body = rest[0] if rest else ""
        first_line = body.lstrip("\n").split("\n", 1)[0] if body else ""
        if not first_line.startswith("| model"):
            continue
        # parse the first table in this block whose first column is `model`
        rows = {}
        lines = body.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("|") and line.startswith("| model"):
                # next must be separator
                if i + 1 >= len(lines):
                    break
                i += 2
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip("|").split("|")]
                    label = cells[0]
                    vals = []
                    for c in cells[1:]:
                        m = re.search(r"([-+]?\d+(?:\.\d+)?)", c)
                        if m:
                            vals.append(float(m.group(1)))
                    rows[label] = vals
                    i += 1
                break
            i += 1
        if rows:
            sections[preset] = rows
    # Average over OOD presets
    ood_presets = [p for p in sections
                   if p != "disc_imf_in_distribution_baseline"]
    models = list(next(iter(sections.values())).keys())
    n_values = list(range(len(next(iter(sections.values()))[models[0]])))
    out: dict[str, list[float]] = {}
    for m in models:
        means = []
        for ni in n_values:
            cell_vals = [sections[p][m][ni] for p in ood_presets
                         if ni < len(sections[p][m])]
            means.append(sum(cell_vals) / max(len(cell_vals), 1))
        out[m] = means
    return out


def _load_in_distribution_baseline(cross_n_audit_md: Path) -> dict[str, list[float]]:
    """Same approach but only `disc_imf_in_distribution_baseline`."""
    return _load_single_step_section(cross_n_audit_md, "disc_imf_in_distribution_baseline")


def _load_single_step_section(cross_n_audit_md: Path, preset: str) -> dict[str, list[float]]:
    md = cross_n_audit_md.read_text(encoding="utf-8")
    for block in md.split("### ")[1:]:
        head, *rest = block.split("\n", 1)
        p = head.split(" (")[0].strip().strip("`")
        if p != preset:
            continue
        body = rest[0] if rest else ""
        rows = {}
        for line in body.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] == "model":
                continue
            vals = []
            for c in cells[1:]:
                m = re.search(r"([-+]?\d+(?:\.\d+)?)", c)
                if m:
                    vals.append(float(m.group(1)))
            rows[cells[0]] = vals
        return rows
    return {}


def _load_rollout_headline(cross_n_rollout_md: Path) -> dict[str, list[float]]:
    """Headline table from cross_N_audit.md (rollout, mean err %)."""
    md = cross_n_rollout_md.read_text(encoding="utf-8")
    return _parse_pipe_table(md, "model")


# ── Plots ────────────────────────────────────────────────────────────
def _style_ax(ax, title: str) -> None:
    ax.set_facecolor(THEME["panel"])
    ax.tick_params(colors=THEME["text"], labelsize=10)
    ax.xaxis.label.set_color(THEME["text"])
    ax.yaxis.label.set_color(THEME["text"])
    ax.set_title(title, color=THEME["text"], fontsize=11, pad=8)
    for s in ax.spines.values():
        s.set_edgecolor(THEME["spine"])
    ax.grid(True, axis="y", which="both", color=THEME["grid"],
            linewidth=0.5, alpha=0.7)


def plot_eval_benchmark(cross_n_audit_md: Path,
                        cross_n_rollout_md: Path,
                        latency_json: Path,
                        out_dir: Path) -> None:
    """Four-panel bar chart: in-distribution baseline + OOD mean err %
    + latency + rollout stability, all per N, with MLP/LSTM/GNN bars."""
    in_dist = _load_in_distribution_baseline(cross_n_audit_md)
    ood     = _load_single_step_ood_mean(cross_n_audit_md)
    rollout = _load_rollout_headline(cross_n_rollout_md)
    lat     = json.loads(latency_json.read_text(encoding="utf-8"))

    N_VALUES = [10, 25, 50, 100]
    base_models = ["MLP", "LSTM", "GNN"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.patch.set_facecolor(THEME["bg"])

    # Panel 1: in-distribution baseline (single-step mean err %)
    ax = axes[0, 0]
    x = np.arange(len(base_models))
    width = 0.18
    for ni, n in enumerate(N_VALUES):
        vals = [in_dist.get(m, [0]*4)[ni] for m in base_models]
        bars = ax.bar(x + (ni - 1.5) * width, vals, width,
                      color=[MODEL_COLORS[m] for m in base_models],
                      edgecolor=THEME["bg"], alpha=0.85, label=f"N={n}")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", color=THEME["text"], fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(base_models)
    ax.set_yscale("log")
    ax.set_ylabel("mean err %  (log)")
    ax.legend(title="training budget", fontsize=8, title_fontsize=9,
              labelcolor=THEME["text"], facecolor=THEME["bg"],
              edgecolor=THEME["spine"])
    _style_ax(ax, "In-distribution baseline  (single-step mean err %)")

    # Panel 2: OOD mean err % (single-step)
    ax = axes[0, 1]
    for ni, n in enumerate(N_VALUES):
        vals = [ood.get(m, [0]*4)[ni] for m in base_models]
        bars = ax.bar(x + (ni - 1.5) * width, vals, width,
                      color=[MODEL_COLORS[m] for m in base_models],
                      edgecolor=THEME["bg"], alpha=0.85, label=f"N={n}")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", color=THEME["text"], fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(base_models)
    ax.set_yscale("log")
    ax.set_ylim(top=320)
    ax.set_ylabel("mean err %  (log)")
    ax.legend(title="training budget", fontsize=8, title_fontsize=9,
              labelcolor=THEME["text"], facecolor=THEME["bg"],
              edgecolor=THEME["spine"], loc="upper left")
    _style_ax(ax, "Solar-System OOD  (single-step mean err %)")

    # Panel 3: latency (single-frame surrogate, log)
    ax = axes[1, 0]
    ssf = lat.get("surrogate_single_frame", {})
    x_lat = np.arange(len(N_VALUES))
    width2 = 0.25
    for mi, m in enumerate(["mlp", "lstm", "gnn"]):
        vals = [ssf.get(m, {}).get(str(n), 0.0) for n in N_VALUES]
        ax.bar(x_lat + (mi - 1) * width2, vals, width2,
               color=[MODEL_COLORS[m.upper()] for _ in vals],
               edgecolor=THEME["bg"], alpha=0.85,
               label=MODEL_SHORT[m.upper()])
    solver_vals = [lat.get("solver", {}).get(str(n), 0.0) for n in N_VALUES]
    ax.plot(x_lat, solver_vals, color=THEME["text"], lw=1.6, ls="--",
            marker="o", markersize=6, label="solver (leapfrog)")
    ax.set_xticks(x_lat)
    ax.set_xticklabels([f"N={n}" for n in N_VALUES])
    ax.set_yscale("log")
    ax.set_ylabel("ms / frame (log)")
    ax.legend(fontsize=8, labelcolor=THEME["text"],
              facecolor=THEME["bg"], edgecolor=THEME["spine"])
    _style_ax(ax, "Latency vs leapfrog  (single-frame, CPU)")

    # Panel 4: rollout stability (mean err % on OOD rollout)
    ax = axes[1, 1]
    for ni, n in enumerate(N_VALUES):
        vals = [rollout.get(m, [0]*4)[ni] for m in base_models]
        bars = ax.bar(x + (ni - 1.5) * width, vals, width,
                      color=[MODEL_COLORS[m] for m in base_models],
                      edgecolor=THEME["bg"], alpha=0.85, label=f"N={n}")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", color=THEME["text"], fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(base_models)
    ax.set_yscale("log")
    ax.set_ylabel("mean err %  (log)")
    ax.legend(title="training budget", fontsize=8, title_fontsize=9,
              labelcolor=THEME["text"], facecolor=THEME["bg"],
              edgecolor=THEME["spine"])
    _style_ax(ax, "OOD rollout stability  (mean err %)")

    fig.suptitle("3D N-body surrogate — 4-metric benchmark",
                 color=THEME["text"], fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "eval_benchmark.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_stability_per_n(single_step: dict[str, list[float]],
                         stable_step: dict[str, list[float]],
                         out_dir: Path) -> None:
    """Per-N stability comparison: single vs stable variant mean err %
    on OOD rollouts. Builds a 3-panel figure (MLP/LSTM/GNN, MSE-style
    curves as a function of N) and one overview."""
    base_models = ["MLP", "LSTM", "GNN"]
    N_VALUES = [10, 25, 50, 100]

    # Per-N 3-panel (one bar chart per N, single vs stable per family)
    for ni, n in enumerate(N_VALUES):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(THEME["bg"])
        x = np.arange(len(base_models))
        width = 0.35
        single_vals = [single_step.get(m, [0]*4)[ni] for m in base_models]
        stable_vals = [stable_step.get(f"{m}_stable", [0]*4)[ni] for m in base_models]
        ax.bar(x - width/2, single_vals, width,
               color=[MODEL_COLORS[m] for m in base_models],
               edgecolor=THEME["bg"], alpha=0.85, label="single-step ckpt")
        ax.bar(x + width/2, stable_vals, width,
               color=[MODEL_COLORS[f"{m}_stable"] for m in base_models],
               edgecolor=THEME["bg"], alpha=0.85, label="stable ckpt")
        for xi, v in enumerate(single_vals):
            ax.annotate(f"{v:.1f}", xy=(xi - width/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", color=THEME["text"], fontsize=8)
        for xi, v in enumerate(stable_vals):
            ax.annotate(f"{v:.1f}", xy=(xi + width/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", color=THEME["text"], fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(base_models)
        ax.set_yscale("log")
        ax.set_ylabel("OOD mean err %  (log)")
        ax.legend(fontsize=10, labelcolor=THEME["text"],
                  facecolor=THEME["bg"], edgecolor=THEME["spine"])
        _style_ax(ax, f"Stability training effect, N = {n}")
        fig.tight_layout()
        out = out_dir / f"stability_N{n}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [plot] {out}")

    # Overview: gradient of (stable / single - 1) across N per family
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(THEME["bg"])
    for m in base_models:
        single = np.array([single_step.get(m, [0]*4)[ni] for ni in range(4)])
        stable = np.array([stable_step.get(f"{m}_stable", [0]*4)[ni] for ni in range(4)])
        delta_pct = 100.0 * (stable - single) / np.maximum(single, 1e-6)
        ax.plot(N_VALUES, delta_pct, marker="o", lw=2.0,
                color=MODEL_COLORS[m], label=m)
        for ni, dv in enumerate(delta_pct):
            ax.annotate(f"{dv:+.0f}%", xy=(N_VALUES[ni], dv),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", color=MODEL_COLORS[m], fontsize=9)
    ax.axhline(0, color=THEME["text"], lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("training budget N")
    ax.set_ylabel("stable − single  (% of single)")
    ax.set_xticks(N_VALUES)
    ax.legend(fontsize=10, labelcolor=THEME["text"],
              facecolor=THEME["bg"], edgecolor=THEME["spine"])
    _style_ax(ax, "Stability-training effect  (positive = stable is worse)")
    fig.tight_layout()
    out = out_dir / "stability_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {out}")


def plot_scaling_latency(latency_json: Path, out_dir: Path) -> None:
    lat = json.loads(latency_json.read_text(encoding="utf-8"))
    N_VALUES = [10, 25, 50, 100, 200]

    ssf = lat["surrogate_single_frame"]
    sba = lat["surrogate_batched_amortised"]
    solver = lat["solver"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(THEME["bg"])

    for m in ["mlp", "lstm", "gnn"]:
        y = [ssf.get(m, {}).get(str(n), 0.0) for n in N_VALUES]
        ax1.plot(N_VALUES, y, marker="o", lw=2.0,
                 color=MODEL_COLORS[m.upper()], label=m.upper())
    y_solver = [solver.get(str(n), 0.0) for n in N_VALUES]
    ax1.plot(N_VALUES, y_solver, marker="o", lw=2.0, ls="--",
             color=THEME["text"], label="solver")
    ax1.set_yscale("log"); ax1.set_xscale("log")
    ax1.set_xlabel("N (body count, log)")
    ax1.set_ylabel("ms / frame (log)")
    ax1.set_xticks(N_VALUES); ax1.set_xticklabels([str(n) for n in N_VALUES])
    ax1.legend(fontsize=10, labelcolor=THEME["text"],
               facecolor=THEME["bg"], edgecolor=THEME["spine"])
    _style_ax(ax1, "Single-frame latency  (CPU)")

    for m in ["mlp", "lstm", "gnn"]:
        y = [sba.get(m, {}).get(str(n), 0.0) for n in N_VALUES]
        ax2.plot(N_VALUES, y, marker="o", lw=2.0,
                 color=MODEL_COLORS[m.upper()], label=m.upper())
    ax2.plot(N_VALUES, y_solver, marker="o", lw=2.0, ls="--",
             color=THEME["text"], label="solver")
    ax2.set_yscale("log"); ax2.set_xscale("log")
    ax2.set_xlabel("N (body count, log)")
    ax2.set_ylabel("ms / frame  amortised (log)")
    ax2.set_xticks(N_VALUES); ax2.set_xticklabels([str(n) for n in N_VALUES])
    ax2.legend(fontsize=10, labelcolor=THEME["text"],
               facecolor=THEME["bg"], edgecolor=THEME["spine"])
    _style_ax(ax2, "Batched amortised latency  (CPU)")

    fig.suptitle("Latency scaling vs N  (5 repeats, CPU-only)",
                 color=THEME["text"], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "scaling_latency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-n-single-step-md",
                    default="real_case_validation/cross_N_audit_single_step.md")
    ap.add_argument("--cross-n-rollout-md",
                    default="real_case_validation/cross_N_audit.md")
    ap.add_argument("--latency-json",
                    default="results/latency_bench.json")
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cross_n_ss = Path(args.cross_n_single_step_md)
    cross_n_ro = Path(args.cross_n_rollout_md)
    lat = Path(args.latency_json)
    if not cross_n_ss.is_file():
        raise SystemExit(f"missing: {cross_n_ss}")
    if not cross_n_ro.is_file():
        raise SystemExit(f"missing: {cross_n_ro}")
    if not lat.is_file():
        raise SystemExit(f"missing: {lat}")

    # ── eval_benchmark.png  (4-panel) ────────────────────────────────
    print("eval_benchmark.png:")
    plot_eval_benchmark(cross_n_ss, cross_n_ro, lat, out_dir)

    # ── stability_*.png  (per-N + overview) ──────────────────────────
    print("Stability per-N + overview:")
    # single-step OOD mean (the 6-model table) — but for stability we
    # want OOD single-step, NOT in-distribution (which is what the
    # ckpts were *trained* on). Use OOD mean.
    ood = _load_single_step_ood_mean(cross_n_ss)
    # The "stable" variant is the *_stable keys in the same dict.
    single_step = {m: ood.get(m, [0]*4) for m in ("MLP", "LSTM", "GNN")}
    stable_step = {f"{m}_stable": ood.get(f"{m}_stable", [0]*4)
                   for m in ("MLP", "LSTM", "GNN")}
    plot_stability_per_n(single_step, stable_step, out_dir)

    # ── scaling_latency.png ──────────────────────────────────────────
    print("Scaling latency:")
    plot_scaling_latency(lat, out_dir)


if __name__ == "__main__":
    main()