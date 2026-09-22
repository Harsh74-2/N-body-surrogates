#!/usr/bin/env python3
"""make_error_tables.py — full error-percentage grid from the 2026-09-22
post-retrain artifacts (nothing hand-transcribed).

Reads:
  rollout     : real_case_validation/report_N{n}/preset_{p}/summary.json
                (calibrated mean_err_pct when present, else
                mean_error_over_L * 100)
  single-step : real_case_validation/report_N{n}/single_step/preset_{p}/
                ss_summary.json (mean_err_pct; stable variants not in the
                OOD single-step dump -- they render as "--")

Writes results/error_tables.md and prints it to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

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
SHORT = {
    "disc_imf_in_distribution_baseline": "disc (in-dist.)",
    "full_solar_system": "full solar system",
    "inner_planets": "inner planets",
    "jupiter_galileans": "jupiter galileans",
    "solar_system_extended": "solar extended",
    "sun_earth_only": "sun-earth only",
    "sun_planets_moon": "sun-planets-moon",
}
N_VALUES = [10, 25, 50, 100]
VARIANTS = ["MLP", "MLP_stable", "LSTM", "LSTM_stable", "GNN", "GNN_stable"]
LABEL = {
    "MLP": "MLP", "MLP_stable": "MLP (stable)",
    "LSTM": "LSTM", "LSTM_stable": "LSTM (stable)",
    "GNN": "GNN", "GNN_stable": "GNN (stable)",
}


def rollout_pct(n: int, preset: str, variant: str) -> float | None:
    f = RCV / f"report_N{n}" / f"preset_{preset}" / "summary.json"
    if not f.exists():
        return None
    pm = json.loads(f.read_text(encoding="utf-8"))["per_model"].get(variant)
    if pm is None:
        return None
    cal = pm.get("calibrated")
    if cal is not None:
        return cal["mean_err_pct"]
    return pm["mean_error_over_L"] * 100.0


def ss_pct(n: int, preset: str, variant: str) -> float | None:
    f = RCV / f"report_N{n}" / "single_step" / f"preset_{preset}" / "ss_summary.json"
    if not f.exists():
        return None
    pm = json.loads(f.read_text(encoding="utf-8"))["per_model"].get(variant)
    return None if pm is None else pm["mean_err_pct"]


def fmt(v: float | None) -> str:
    if v is None:
        return "&mdash;"
    if v > 100.0:
        return f"div. ({v:.0f}%)"
    return f"{v:.1f}%"


def table(kind: str, cell_fn) -> str:
    head = ("### Autoregressive rollout (mean error, % of scale length L)"
            if kind == "rollout"
            else "### Single-step (mean error, % of scale length L)")
    lines = [head, "",
             "| N | variant | " + " | ".join(SHORT[p] for p in PRESETS) + " | OOD mean |",
             "|---|---|" + "---|" * (len(PRESETS) + 1)]
    for n in N_VALUES:
        for v in VARIANTS:
            vals = [cell_fn(n, p, v) for p in PRESETS]
            ood = vals[1:]
            kept = [x for x in ood if x is not None and x <= 100.0]
            mean = sum(kept) / len(kept) if kept else None
            lines.append(
                f"| {n} | {LABEL[v]} | " + " | ".join(fmt(x) for x in vals)
                + f" | {fmt(mean)} |")
    return "\n".join(lines)


def main() -> None:
    md = "\n\n".join([
        "# Error-percentage grid — 2026-09-22 post-retrain",
        "",
        "Every number is read directly from the per-preset JSON artifacts of",
        "the fresh retrain (`real_case_validation/report_N{n}/...`). Rollout",
        "values use the calibrated post-hoc error where present. Cells above",
        "100% of L are marked div. and excluded from the OOD mean; the disc",
        "baseline is in-distribution and excluded from the OOD mean.",
        table("rollout", rollout_pct),
        table("single-step", ss_pct),
    ]) + "\n"
    out = REPO / "results" / "error_tables.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[md] -> {out}")


if __name__ == "__main__":
    main()