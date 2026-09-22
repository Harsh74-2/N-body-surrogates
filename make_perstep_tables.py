#!/usr/bin/env python3
"""make_perstep_tables.py — per-step autoregressive error grid from the
local per-step dump runs (real_case_validation/_perstep_N{n}/).

For each (N, preset, variant) the dumped rollout arrays give the error at
every frame after the warm-up window closes; this script samples that
curve at k = 1, 2, 4, 8, 16, 32, 64 and the final frame, as mean
positional error in % of the preset scale length L (dimensionless L=1,
so % = mean over bodies of |pred_pos - book_pos| x 100).

Reference = `book_pos.npy` (the closed-form Kepler/"book" reference, the
same reference the calibrated lane uses) -- NOT the leapfrog reference,
so full-window means sit ~2% below the uncalibrated summary.json values,
which compare against the leapfrog reference instead.

Writes results/perstep_tables.md and prints it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
RCV = REPO / "real_case_validation"
PRESETS = [
    "disc_imf_in_distribution_baseline",
    "full_solar_system",
    "inner_planets",
    "jupiter_galileans",
    "solar_system_extended",
    "sun_earth_only",
    "sun_planets_moon",
]
N_VALUES = [10, 25, 50, 100]
VARIANTS = ["MLP", "MLP_stable", "LSTM", "LSTM_stable", "GNN", "GNN_stable"]
LABEL = {
    "MLP": "MLP", "MLP_stable": "MLP (stable)",
    "LSTM": "LSTM", "LSTM_stable": "LSTM (stable)",
    "GNN": "GNN", "GNN_stable": "GNN (stable)",
}
KS = [1, 2, 4, 8, 16, 32, 64, 128]


def curve(n: int, preset: str, variant: str) -> np.ndarray | None:
    """Mean-over-bodies positional error vs book, per frame (dimensionless)."""
    d = RCV / f"_perstep_N{n}" / f"preset_{preset}"
    meta_f = d / "preds_meta.json"
    if not meta_f.exists():
        return None
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    if variant not in meta["models"]:
        return None
    preds = np.load(d / "preds.npy")
    book = np.load(d / "book_pos.npy")
    i = meta["models"].index(variant)
    err = np.linalg.norm(preds[i, :, :, :3] - book[:, :, :], axis=-1)  # (T, N)
    return err.mean(axis=1)  # (T,) per-frame mean over bodies


def fmt(v: float) -> str:
    return f"{v * 100:.1f}" if v < 10.0 else f"{v * 100:.0f}"


def main() -> None:
    lines = [
        "# Per-step autoregressive error — 2026-09-22 post-retrain",
        "",
        "Mean positional error (% of scale length L) at rollout step k",
        "after the warm-up window closes, averaged over bodies. Reference",
        "= the closed-form Kepler/book reference (same as the calibrated",
        "lane). k=1 is effectively the single-step protocol; the final",
        "column is the last frame of the rollout window.",
    ]

    # --- per-preset tables ---
    for preset in PRESETS:
        tag = "in-distribution" if preset == PRESETS[0] else "OOD"
        lines += ["", f"## {preset} ({tag})", ""]
        ks_avail = []
        rows = []
        for n in N_VALUES:
            for v in VARIANTS:
                c = curve(n, preset, v)
                if c is None:
                    continue
                T = len(c)
                ks = [k for k in KS if k < T] + [T - 1]
                if not ks_avail:
                    ks_avail = ks
                rows.append((n, v, c, ks, T))
        if not rows:
            continue
        header_ks = ks_avail
        lines.append("| N | variant | " + " | ".join(f"k={k}" for k in header_ks)
                     + " |")
        lines.append("|---|---|" + "---|" * len(header_ks))
        for n, v, c, ks, T in rows:
            cells = [fmt(c[min(k, T - 1)]) for k in header_ks]
            lines.append(f"| {n} | {LABEL[v]} | " + " | ".join(cells) + " |")

    # --- OOD-mean per-step table (mean over the 6 OOD presets, per step) ---
    lines += ["", "## OOD mean per step (mean over the six OOD presets)", ""]
    header = None
    rows = []
    for n in N_VALUES:
        for v in VARIANTS:
            curves = [curve(n, p, v) for p in PRESETS[1:]]
            curves = [c for c in curves if c is not None]
            if not curves:
                continue
            T = min(len(c) for c in curves)
            mean_c = np.mean([c[:T] for c in curves], axis=0)
            ks = [k for k in KS if k < T] + [T - 1]
            if header is None:
                header = ks
            rows.append((n, v, mean_c, ks))
    if header:
        lines.append("| N | variant | " + " | ".join(f"k={k}" for k in header) + " |")
        lines.append("|---|---|" + "---|" * len(header))
        for n, v, c, ks in rows:
            cells = [fmt(c[min(k, len(c) - 1)]) for k in header]
            lines.append(f"| {n} | {LABEL[v]} | " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    out = REPO / "results" / "perstep_tables.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[md] -> {out}")


if __name__ == "__main__":
    main()