#!/usr/bin/env python3
"""
predictive_horizon_report.py — the "useful horizon" table for the thesis.

Reads the per-step model/persistence ratio curves the Stage-4 stability
benchmark already recorded (results/N{n}/stability.json ->
identity_baseline.model_over_identity) and reports, per cell:

  * predictive horizon k* — the first rollout step at which the surrogate's
    cumulative rollout MSE exceeds the frozen-anchor (persistence) baseline;
    i.e. the last step at which the model is still the better predictor.
    Cells that never cross within the horizon report "> K".
  * ratio snapshots at k = 32 / 64 / K for context.

Chaos-theory framing: k* is the surrogate's predictive horizon, the
autoregressive analogue of a Lyapunov-time-limited forecast range (Gemini
review, 2026-09-22 — endorsed as a thesis table).

Usage:
    python predictive_horizon_report.py              # default results/ root
    python predictive_horizon_report.py --results-root results --N 10 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANT_ORDER = ["single_step", "stable"]
MODEL_ORDER = ["mlp", "lstm", "gnn"]


def horizon_of(ratio: list) -> tuple:
    """(k*, crossed?, ratio@last) from a per-step ratio list (None = NaN)."""
    r = np.array(ratio, dtype=float)
    steps = r.size
    above = np.where(r >= 1.0)[0]
    if above.size:
        return int(above[0]) + 1, True, float(r[above[0]])
    return None, False, float(r[-1]) if np.isfinite(r[-1]) else float("nan")


def snap(ratio: list, k: int) -> float:
    r = np.array(ratio, dtype=float)
    if r.size >= k:
        v = float(r[k - 1])
        return v if np.isfinite(v) else float("nan")
    return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--N", type=int, nargs="+", default=[10, 25, 50, 100])
    args = p.parse_args()
    root = Path(args.results_root)

    rows, cells = [], []
    for n in args.N:
        path = root / f"N{n}" / "stability.json"
        if not path.is_file():
            print(f"[skip] {path} missing")
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            ratio = rec["identity_baseline"]["model_over_identity"]
            k_star, crossed, r_at_cross = horizon_of(ratio)
            steps = len(ratio)
            row = {
                "cell": f"N{n}/{rec['model_type']}_{rec['variant']}",
                "N": n,
                "model": rec["model_type"],
                "variant": rec["variant"],
                "K": steps,
                "predictive_horizon": k_star if crossed else None,
                "never_crosses_within_K": not crossed,
                "ratio_at_crossover": None if not crossed else r_at_cross,
                "ratio_k32": snap(ratio, 32),
                "ratio_k64": snap(ratio, 64),
                "ratio_k128": snap(ratio, 128),
            }
            rows.append(row)
            cells.append(
                f"N{n:<3d} {rec['model_type']:4s} {rec['variant']:12s} "
                f"K* = {'>' + str(steps) if not crossed else str(k_star):>4s}"
                f"   ratio@32={row['ratio_k32']:.3f}"
                f"  ratio@64={row['ratio_k64']:9.3f}"
                f"  ratio@128={row['ratio_k128']:11.3f}")
            print(cells[-1])

    out_json = root / "predictive_horizon.json"
    out_md = root / "predictive_horizon.md"
    out_json.write_text(json.dumps(
        {"description": "Predictive horizon k* = first rollout step where "
                        "model rollout MSE exceeds the persistence anchor "
                        "('>' + K = never crosses within the horizon).",
         "cells": rows}, indent=2), encoding="utf-8")

    md = ["# Predictive horizon vs persistence (K=128 rollout benchmark)",
          "",
          "k* = first rollout step at which the surrogate's cumulative "
          "rollout MSE exceeds the frozen-anchor persistence baseline "
          "(the last step at which the model is the better predictor).",
          "`> 128` = never crosses within the benchmark horizon.",
          "",
          "| cell | K* | ratio@32 | ratio@64 | ratio@128 |",
          "|---|---|---|---|---|"]
    for row in rows:
        k_disp = ("> 128" if row["never_crosses_within_K"]
                  else str(row["predictive_horizon"]))
        md.append(
            f"| {row['cell']} | {k_disp} "
            f"| {row['ratio_k32']:.3f} | {row['ratio_k64']:.3f} "
            f"| {row['ratio_k128']:.3f} |")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[json] -> {out_json}\n[md]   -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())