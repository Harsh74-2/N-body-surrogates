#!/usr/bin/env python3
"""regen_thesis_figures.py — regenerate the four stale thesis figure
families from the FINAL 2026-09-22 post-retrain artifacts:

  thesis_figures/eval_benchmark_N{10,25,50,100}.png
      4-panel per-N benchmark (MSE, |dE/E0|, latency, K=50 rollout drift)
      from results/all_eval.json + results/latency_bench.json.
  thesis_figures/loss_curves_N{10,25,50,100}.png
      3x2 grid of train/val total-loss curves from
      training_runs/N*/{variant}/history.json.
  thesis_figures/dashboard.png
      Final test-set MSE per architecture and body count + per-architecture
      validation-loss overlay (N=25), from all_eval.json + history.json.
  thesis_figures/param_count.png
      Trainable-parameter bars from all_eval.json n_params.

Theme matches regen_top_level_plots.py so figures sit together.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "thesis_figures"

THEME = {
    "bg": "#11151c", "panel": "#161c26", "grid": "#3a4658",
    "text": "#e7ecf2", "spine": "#7d8597", "violet": "#9b6dff",
    "good": "#3ddc97", "accent": "#56b6f2",
}
# (label, json variant key under training_runs/<N>/, color, all_eval macro base)
VARIANTS = [
    ("MLP — single-step", "mlp", "#9b6dff", "Mlp"),
    ("MLP — stable", "mlp_stable", "#c79bff", "MlpSt"),
    ("LSTM — single-step", "lstm", "#3ddc97", "Lstm"),
    ("LSTM — stable", "lstm_stable", "#9af3c8", "LstmSt"),
    ("GNN — single-step", "gnn", "#56b6f2", "Gnn"),
    ("GNN — stable", "gnn_stable", "#9bd5ff", "GnnSt"),
]
NS = [10, 25, 50, 100]


def style_ax(ax):
    ax.set_facecolor(THEME["panel"])
    ax.figure.set_facecolor(THEME["bg"])
    for s in ax.spines.values():
        s.set_color(THEME["spine"])
    ax.tick_params(colors=THEME["text"], which="both")
    ax.xaxis.label.set_color(THEME["text"])
    ax.yaxis.label.set_color(THEME["text"])
    ax.title.set_color(THEME["text"])
    ax.grid(color=THEME["grid"], alpha=0.4, which="both")


def load_all_eval():
    cells = {}
    for c in json.loads((REPO / "results" / "all_eval.json")
                        .read_text(encoding="utf-8")):
        n, mv = c["cell"].split("/")
        key = mv if c["variant"] == "single_step" else mv + "_stable"
        cells[(int(n[1:]), key)] = c
    return cells


def bar_labels(ax, bars, fmt="{:.2e}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8,
                    color=THEME["text"], rotation=0)


# --------------------------------------------------------------- per-N 4-panel
def eval_benchmark_per_n(n, cells, bench):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"N = {n}  ·  3D N-body surrogate — four-metric benchmark",
                 color=THEME["text"], fontsize=14, fontweight="bold")
    labels = [v[0].replace(" — ", "\n") for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]

    # (a) test MSE single-step, log, with identity floor
    ax = axes[0][0]
    vals = [cells[(n, v[1])]["mse"] for v in VARIANTS]
    ident = [cells[(n, v[1])]["mse_identity"] for v in VARIANTS]
    bars = ax.bar(labels, vals, color=colors)
    for x, iv in zip(range(len(vals)), ident):
        ax.plot([x - 0.4, x + 0.4], [iv, iv], ls="--", lw=1.2,
                color=THEME["spine"])
    ax.set_yscale("log")
    ax.set_ylabel("MSE single-step")
    ax.set_title("MSE single-step (↓, dashed = identity floor)", fontsize=10)
    style_ax(ax)
    bar_labels(ax, bars)

    # (b) single-step energy error
    ax = axes[0][1]
    vals = [cells[(n, v[1])]["energy"] for v in VARIANTS]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("|ΔE / E₀| single-step")
    ax.set_title("|ΔE / E₀| single-step (↓)", fontsize=10)
    style_ax(ax)
    bar_labels(ax, bars)

    # (c) single-frame latency vs solver
    ax = axes[1][0]
    models = [("MLP", "mlp", "#9b6dff"), ("LSTM", "lstm", "#3ddc97"),
              ("GNN", "gnn", "#56b6f2")]
    vals = [bench["surrogate_single_frame"][m][str(n)] for _, m, _ in models]
    bars = ax.bar([m[0] for m in models], vals,
                  color=[m[2] for m in models])
    solver = bench["solver"][str(n)]
    ax.axhline(solver, ls="--", lw=1.5, color=THEME["text"])
    ax.annotate(f"solver {solver:.3g} ms", (0.02, solver),
                xycoords=("axes fraction", "data"), va="bottom",
                fontsize=8, color=THEME["text"])
    ax.set_yscale("log")
    ax.set_ylabel("ms / frame (CPU, single)")
    ax.set_title("Latency vs leapfrog (single-frame, CPU, ↓)", fontsize=10)
    style_ax(ax)
    bar_labels(ax, bars, "{:.3g}")

    # (d) K=50 rollout drift
    ax = axes[1][1]
    vals = [cells[(n, v[1])]["rollout"] for v in VARIANTS]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("K=50 rollout drift (MSE)")
    ax.set_title("Rollout stability, K=50 (↓)", fontsize=10)
    style_ax(ax)
    bar_labels(ax, bars)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / f"eval_benchmark_N{n}.png"
    fig.savefig(out, dpi=150, facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  -> {out.name}")


# --------------------------------------------------------------- loss curves
def loss_curves(n):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Training loss curves, N = {n} "
                 "(3 architectures × 2 variants)",
                 color=THEME["text"], fontsize=16, fontweight="bold")
    for ax, (label, key, color, _) in zip(axes.flat, VARIANTS):
        h = json.loads((REPO / "training_runs" / f"N{n}" / key /
                        "history.json").read_text(encoding="utf-8"))
        hist = h["history"]
        ep = [r["epoch"] for r in hist]
        tr = [r["train_total"] if "train_total" in r
              else r["train_components"]["total"] for r in hist]
        va = [r["val_total"] if "val_total" in r
              else r["val_components"]["total"] for r in hist]
        ax.plot(ep, tr, "-", lw=1.6,
                color=color, label="train")
        ax.plot(ep, va, "--", lw=1.4,
                color=color, alpha=0.75, label="val")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel("total loss", fontsize=9)
        ax.legend(facecolor=THEME["panel"], labelcolor=THEME["text"],
                  edgecolor=THEME["spine"])
        style_ax(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / f"loss_curves_N{n}.png"
    fig.savefig(out, dpi=130, facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  -> {out.name}")


# ------------------------------------------------------------------ dashboard
def dashboard(cells):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Validation dashboard — final test-set MSE per architecture "
                 "and body count", color=THEME["text"], fontsize=14,
                 fontweight="bold")
    # (a) grouped bars: x = architecture, series = N
    ax = axes[0]
    shades = {10: "#9b6dff", 25: "#b58cff", 50: "#c9aaff", 100: "#d9c4ff"}
    xs = np.arange(3)
    w = 0.2
    for i, n in enumerate(NS):
        vals = [cells[(n, m)]["mse"] for m in ("mlp", "lstm", "gnn")]
        bars = ax.bar(xs + (i - 1.5) * w, vals, w, label=f"N={n}",
                      color=shades[n])
    for i, n in enumerate(NS):
        for j, m in enumerate(("mlp", "lstm", "gnn")):
            iv = cells[(n, m)]["mse_identity"]
            x = j + (i - 1.5) * w
            ax.plot([x - w / 2, x + w / 2], [iv, iv], ls="--", lw=1.1,
                    color=THEME["spine"], zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["MLP", "LSTM", "GNN"])
    ax.set_ylabel("test MSE (log)")
    ax.set_title("Final test-set MSE (dashed = identity floor)", fontsize=10)
    ax.legend(facecolor=THEME["panel"], labelcolor=THEME["text"],
              edgecolor=THEME["spine"], title="training budget")
    style_ax(ax)

    # (b) per-architecture validation-loss overlay at N=25
    ax = axes[1]
    for label, key, color, _ in VARIANTS:
        h = json.loads((REPO / "training_runs" / "N25" / key /
                        "history.json").read_text(encoding="utf-8"))
        hist = h["history"]
        ep = [r["epoch"] for r in hist]
        va = [r["val_total"] if "val_total" in r
              else r["val_components"]["total"] for r in hist]
        ls = "-" if not key.endswith("_stable") else "--"
        ax.plot(ep, va, ls, lw=1.5, color=color,
                label=label.replace(" — ", " "))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation total loss (log)")
    ax.set_title("Per-architecture validation loss, N=25 "
                 "(solid = single-step, dashed = stable)", fontsize=10)
    ax.legend(facecolor=THEME["panel"], labelcolor=THEME["text"],
              edgecolor=THEME["spine"], fontsize=8)
    style_ax(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "dashboard.png"
    fig.savefig(out, dpi=150, facecolor=THEME["bg"])
    plt.close(fig)
    print("  -> dashboard.png")


# --------------------------------------------------------------- param count
def param_count(cells):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    models = [("MLP", "mlp", "#9b6dff"), ("LSTM", "lstm", "#3ddc97"),
              ("GNN", "gnn", "#56b6f2")]
    vals = [cells[(10, m)]["n_params"] for _, m, _ in models]
    bars = ax.bar([m[0] for m in models], vals,
                  color=[m[2] for m in models])
    ax.set_yscale("log")
    ax.set_ylabel("trainable parameters (log scale)")
    ax.set_title("Trainable parameters — per-body, shared weights "
                 "(N-independent)")
    style_ax(ax)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=THEME["text"])
    fig.tight_layout()
    out = OUT / "param_count.png"
    fig.savefig(out, dpi=150, facecolor=THEME["bg"])
    plt.close(fig)
    print("  -> param_count.png")


# --------------------------------------------------------------- scaling plot
def scaling_plot(bench):
    fig, ax = plt.subplots(figsize=(9, 6))
    ns = sorted(int(k) for k in bench["solver"])
    colors = {"mlp": "#9b6dff", "lstm": "#3ddc97", "gnn": "#56b6f2"}
    labels = {"mlp": "MLP", "lstm": "LSTM", "gnn": "GNN"}
    solver = [bench["solver"][str(n)] for n in ns]
    ax.plot(ns, solver, "-", lw=2.4, color=THEME["text"],
            marker="D", markersize=6, label="leapfrog solver")
    for m in ("mlp", "lstm", "gnn"):
        ssf = [bench["surrogate_single_frame"][m][str(n)] for n in ns]
        bat = [bench["surrogate_batched_amortised"][m][str(n)] for n in ns]
        ax.plot(ns, ssf, "-", lw=1.8, color=colors[m],
                marker="o", markersize=5, label=f"{labels[m]} (single-frame)")
        ax.plot(ns, bat, "--", lw=1.5, color=colors[m],
                marker="s", markersize=4, alpha=0.8,
                label=f"{labels[m]} (batched, B<=64)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("body count $N$")
    ax.set_ylabel("latency per frame (ms, log)")
    ax.set_title("Per-frame latency scaling: solver vs surrogates")
    ax.legend(facecolor=THEME["panel"], labelcolor=THEME["text"],
              edgecolor=THEME["spine"], fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    out = OUT / "scaling_plot.png"
    fig.savefig(out, dpi=150, facecolor=THEME["bg"])
    plt.close(fig)
    print("  -> scaling_plot.png")


# ----------------------------------------------------- aggregate benchmark
def load_macros():
    return json.loads((REPO / "results" / "metrics_macros.json")
                      .read_text(encoding="utf-8"))


def _texnum(s):
    """Parse a metrics.tex-style value ('$7.79\\times10^{-8}$', '11.6\\%',
    'div.', '>100', '73.0\\%') to float or None."""
    s = s.strip().strip("$")
    m = re.match(r'([\d.]+)\\times10\^\{(-?\d+)\}$', s)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    s = s.replace("\\%", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


SHADES = {10: "#9b6dff", 25: "#b58cff", 50: "#c9aaff", 100: "#d9c4ff"}


def eval_benchmark_agg(cells, bench, macros):
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    fig.suptitle("3D N-body surrogate — 4-metric benchmark (all 24 "
                 "checkpoints)", color=THEME["text"], fontsize=15,
                 fontweight="bold")
    arches = [("MLP", "mlp"), ("LSTM", "lstm"), ("GNN", "gnn")]
    st_key = {"mlp": "MlpSt", "lstm": "LstmSt", "gnn": "GnnSt"}
    base = {"mlp": "Mlp", "lstm": "Lstm", "gnn": "Gnn"}
    ntok = {10: "Nten", 25: "Ntwentyfive", 50: "Nfifty", 100: "Nhundred"}

    # (a) in-distribution single-step MSE + EV, all 24 checkpoints
    ax = axes[0][0]
    w = 0.105
    for j, (alab, akey) in enumerate(arches):
        for i, n in enumerate(NS):
            for v, (suffix, key) in enumerate(
                    [("", akey), ("_stable", akey + "_stable")]):
                c = cells[(n, key)]
                k = i * 2 + v
                x = j + (k - 3.5) * w
                b = ax.bar(x, c["mse"], w * 0.94, color=SHADES[n],
                           hatch="///" if v else None,
                           edgecolor=THEME["bg"], linewidth=0.3)
                ax.plot([x - w / 2, x + w / 2],
                        [c["mse_identity"]] * 2, ls="--", lw=0.9,
                        color=THEME["spine"], zorder=4)
                ax.annotate(f"{c['mse']:.1e}", (x, c["mse"]),
                            ha="center", va="bottom", rotation=90,
                            fontsize=5.6, color=THEME["text"],
                            xytext=(0, 1), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[0] for a in arches])
    ax.set_ylabel("single-step test MSE (log)")
    ax.set_title("(a) In-distribution single-step MSE — all 24 checkpoints "
                 "(hatched = stability-trained, dashed = identity floor)",
                 fontsize=9.5)
    style_ax(ax)
    ax2 = ax.twinx()
    for j, (alab, akey) in enumerate(arches):
        for i, n in enumerate(NS):
            for v, key in enumerate([akey, akey + "_stable"]):
                ev = cells[(n, key)]["mse_explained_var"]
                k = i * 2 + v
                ax2.plot(j + (k - 3.5) * w, ev, "D", ms=3.5,
                         color="#ff9f43", alpha=0.9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("explained variance (diamonds, right)", color="#ff9f43")
    ax2.tick_params(colors="#ff9f43", which="both")
    for s in ax2.spines.values():
        s.set_color(THEME["spine"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=SHADES[n], label=f"N={n}")
               for n in NS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#dddddd", hatch="///",
                                 label="stable"))
    ax.legend(handles=handles, facecolor=THEME["panel"],
              labelcolor=THEME["text"], edgecolor=THEME["spine"],
              fontsize=8, loc="upper right")

    # (b) OOD single-step, no-compounding protocol (mean over 6 presets)
    ax = axes[0][1]
    for j, (alab, akey) in enumerate(arches):
        for i, n in enumerate(NS):
            v = _texnum(macros[f"resNoComp{base[akey]}{ntok[n]}"]["tex"])
            x = j + (i - 1.5) * 0.2
            ax.bar(x, v, 0.2, color=SHADES[n], edgecolor=THEME["bg"],
                   linewidth=0.3)
            ax.annotate(f"{v:.1f}", (x, v), ha="center", va="bottom",
                        fontsize=7, color=THEME["text"])
    ax.set_xticks(range(3))
    ax.set_xticklabels([a[0] for a in arches])
    ax.set_ylabel("OOD single-step mean err (%)")
    ax.set_title("(b) OOD single-step mean error, no-compounding protocol "
                 "(mean over 6 presets)", fontsize=9.5)
    style_ax(ax)

    # (c) per-frame latency, solver vs surrogates
    ax = axes[1][0]
    ns_b = sorted(int(k) for k in bench["solver"])
    acol = {"mlp": "#9b6dff", "lstm": "#3ddc97", "gnn": "#56b6f2"}
    for j, (alab, akey) in enumerate(arches):
        vals = [bench["surrogate_single_frame"][akey][str(n)] for n in ns_b]
        ax.bar([j * 5 + t - 1 for t in range(5)], vals, 0.8,
               color=acol[akey], label=alab, edgecolor=THEME["bg"],
               linewidth=0.3)
        for t, v in enumerate(vals):
            ax.annotate(f"{v:.3g}", (j * 5 + t - 1, v), ha="center",
                        va="bottom", fontsize=6, rotation=90,
                        color=THEME["text"])
    solver = [bench["solver"][str(n)] for n in ns_b]
    ax.plot(range(5), solver, "--", lw=2, color=THEME["text"],
            marker="D", ms=5, label="leapfrog solver")
    for t, v in enumerate(solver):
        ax.annotate(f"{v:.3g}", (t, v), ha="center", va="bottom",
                    fontsize=6, color=THEME["text"])
    ax.set_yscale("log")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"N={n}" for n in ns_b])
    ax.set_ylabel("ms / frame (CPU, single, log)")
    ax.set_title("(c) Per-frame latency, solver vs surrogates (B=1)",
                 fontsize=9.5)
    ax.legend(facecolor=THEME["panel"], labelcolor=THEME["text"],
              edgecolor=THEME["spine"], fontsize=8)
    style_ax(ax)

    # (d) OOD rollout mean error, all 24 cells
    ax = axes[1][1]
    vkeys = [("MLP", "Mlp", "mlp"), ("MLP-st", "MlpSt", "mlp_stable"),
             ("LSTM", "Lstm", "lstm"), ("LSTM-st", "LstmSt", "lstm_stable"),
             ("GNN", "Gnn", "gnn"), ("GNN-st", "GnnSt", "gnn_stable")]
    for j, (vlab, mkey, _) in enumerate(vkeys):
        for i, n in enumerate(NS):
            raw = macros[f"resRollOOD{mkey}{ntok[n]}"]["raw"]
            tex = macros[f"resRollOOD{mkey}{ntok[n]}"]["tex"]
            x = j + (i - 1.5) * 0.2
            if raw is None:  # all 6 OOD presets diverged
                ax.bar(x, 100, 0.2, color=SHADES[n], hatch="xxx",
                       edgecolor="#ff6b6b", linewidth=0.8)
                ax.annotate(">100", (x, 100), ha="center", va="bottom",
                            fontsize=7, color="#ff6b6b")
            else:
                ax.bar(x, raw, 0.2, color=SHADES[n], edgecolor=THEME["bg"],
                       linewidth=0.3)
                ax.annotate(f"{raw:.0f}", (x, raw), ha="center",
                            va="bottom", fontsize=6.5, color=THEME["text"])
    ax.set_xticks(range(6))
    ax.set_xticklabels([v[0] for v in vkeys], fontsize=9)
    ax.set_ylabel("OOD rollout mean err (%)")
    ax.set_ylim(0, 112)
    ax.set_title("(d) OOD rollout mean error, K=50 calibrated rollout "
                 "(xxx = all presets diverged)", fontsize=9.5)
    style_ax(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REPO / "plots" / "eval_benchmark.png"
    fig.savefig(out, dpi=150, facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------- stability figures
STAB_VARIANTS = [
    ("MLP (single)", "mlp", "single_step", "#9b6dff", "-"),
    ("MLP (stable)", "mlp", "stable", "#c79bff", "--"),
    ("LSTM (single)", "lstm", "single_step", "#3ddc97", "-"),
    ("LSTM (stable)", "lstm", "stable", "#9af3c8", "--"),
    ("GNN (single)", "gnn", "single_step", "#56b6f2", "-"),
    ("GNN (stable)", "gnn", "stable", "#9bd5ff", "--"),
]


def load_stability(n):
    p = REPO / "results" / f"N{n}" / "stability.json"
    cells = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for c in cells:
        out[(c["model_type"], c["variant"])] = c
    return out


def stability_per_n(n):
    cells = load_stability(n)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(f"Rollout stability, N={n}  (K=128, 10 starts)",
                 color=THEME["text"], fontsize=14, fontweight="bold")
    k = range(len(cells[("mlp", "single_step")]["per_step"]["mse"]))

    # (a) per-step MSE with persistence (identity) floor
    ax = axes[0]
    ident = cells[("mlp", "single_step")]["identity_baseline"]["per_step_mse"]
    ax.plot(k, ident, ":", lw=2.0, color=THEME["text"],
            label="persistence (identity)")
    for label, mt, var, color, ls in STAB_VARIANTS:
        ax.plot(k, cells[(mt, var)]["per_step"]["mse"], ls, lw=1.5,
                color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("rollout step $k$")
    ax.set_ylabel("pos+vel MSE (log)")
    ax.set_title("Rollout MSE vs step (dotted = persistence floor)",
                 fontsize=10)
    ax.legend(facecolor=THEME["panel"], labelcolor=THEME["text"],
              edgecolor=THEME["spine"], fontsize=8)
    style_ax(ax)

    # (b) per-step energy drift
    ax = axes[1]
    for label, mt, var, color, ls in STAB_VARIANTS:
        ax.plot(k, cells[(mt, var)]["per_step"]["energy_drift"], ls,
                lw=1.5, color=color, label=label)
    ax.set_xlabel("rollout step $k$")
    ax.set_ylabel(r"$|\Delta E / E_0|$")
    ax.set_title("Energy drift vs step", fontsize=10)
    style_ax(ax)

    # (c) composed loss
    ax = axes[2]
    for label, mt, var, color, ls in STAB_VARIANTS:
        ax.plot(k, cells[(mt, var)]["per_step"]["loss"], ls, lw=1.5,
                color=color, label=label)
    ax.set_xlabel("rollout step $k$")
    ax.set_ylabel("composed loss")
    ax.set_title("Composed loss vs step", fontsize=10)
    style_ax(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / f"stability_N{n}.png"
    fig.savefig(out, dpi=150, facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  -> {out.name}")


def stability_overview():
    summ = json.loads((REPO / "results" / "stability_summary.json")
                      .read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, mt, var, color, ls in STAB_VARIANTS:
        key = f"{mt}_{'single_step' if var == 'single_step' else 'stable'}"
        ys = [summ[str(n)][key]["mse_slope"] for n in NS]
        ax.plot(NS, ys, ls, lw=1.8, color=color, marker="o", ms=5,
                label=label)
        for n, y in zip(NS, ys):
            ax.annotate(f"{y:.1e}", (n, y), ha="center", va="bottom",
                        fontsize=6.5, color=color,
                        xytext=(0, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("body count $N$")
    ax.set_ylabel("rollout MSE slope (log)")
    ax.set_title("Stability slopes vs body count — solid = single-step, "
                 "dashed = stability-trained")
    ax.legend(facecolor=THEME["panel"], labelcolor=THEME["text"],
              edgecolor=THEME["spine"], fontsize=8)
    style_ax(ax)
    fig.tight_layout()
    out = REPO / "plots" / "stability_overview.png"
    fig.savefig(out, dpi=150, facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  -> {out}")


def main():
    cells = load_all_eval()
    bench = json.loads((REPO / "results" / "latency_bench.json")
                       .read_text(encoding="utf-8"))
    for n in NS:
        eval_benchmark_per_n(n, cells, bench)
        loss_curves(n)
    dashboard(cells)
    param_count(cells)
    scaling_plot(bench)
    for n in NS:
        stability_per_n(n)
    stability_overview()


if __name__ == "__main__":
    main()