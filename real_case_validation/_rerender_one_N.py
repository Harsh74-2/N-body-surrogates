"""
One-off regen helper for the supervisor's per-preset trajectory /
energy / error-vs-reference plots after the dashed=ref / solid=predicted
convention change.

For one training N, runs the runner for every preset so the trajectory
PNG (and the energy / error-vs-reference PNG) inside each
`report_N{n}/preset_<name>/` dir reflects the new convention. The
runner is inference-only (no retraining), CPU-compatible, and reuses
the existing ckpts under `training_runs/N{n}/<arch>/model_best.pt`.

Usage:
    python -m real_case_validation._rerender_one_N --n 25
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PRESETS_ORDERED = [
    "sun_earth_only",
    "jupiter_galileans",
    "sun_planets_moon",
    "inner_planets",
    "full_solar_system",
    "solar_system_extended",
    "disc_imf_in_distribution_baseline",
]

VARIANTS = [
    ("gnn",  "gnn",  "GNN"),
    ("gnn_stable",  "gnn",  "GNN_stable"),
    ("lstm", "lstm", "LSTM"),
    ("lstm_stable", "lstm", "LSTM_stable"),
    ("mlp",  "mlp",  "MLP"),
    ("mlp_stable",  "mlp",  "MLP_stable"),
]


def _ckpt_arg(n: int, surr_type: str, human: str) -> str:
    ckpt = REPO_ROOT / "training_runs" / f"N{n}" / surr_type / "model_best.pt"
    rel = str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/")
    return f"--ckpt {rel}:{surr_type}:{human}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True,
                    help="Training body count N ∈ {10, 25, 50, 100}.")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "real_case_validation" / f"report_N{args.n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for preset in PRESETS_ORDERED:
        ckpt_args = []
        present_variants = []
        for sub, surr, human in VARIANTS:
            ckpt = REPO_ROOT / "training_runs" / f"N{args.n}" / sub / "model_best.pt"
            if not ckpt.is_file():
                print(f"  [skip] {ckpt}", flush=True)
                continue
            ckpt_args += ["--ckpt",
                          f"{str(ckpt.relative_to(REPO_ROOT)).replace(chr(92), '/')}"
                          f":{surr}:{human}"]
            present_variants.append(human)
        if not ckpt_args:
            failures.append((preset, "no ckpts"))
            continue
        cmd = [sys.executable, "-m",
               "real_case_validation.real_case_runner",
               "--preset", preset,
               "--out", str(out_dir),
               "--quick"] + ckpt_args
        print(f"\n[preset] {preset}  ({len(present_variants)} ckpts)",
              flush=True)
        try:
            r = subprocess.run(cmd, cwd=REPO_ROOT,
                               capture_output=True, text=True,
                               timeout=1200)
            if r.returncode != 0:
                print(f"  [fail] exit {r.returncode}", flush=True)
                print(r.stderr[-1500:], flush=True)
                failures.append((preset, f"exit {r.returncode}"))
        except subprocess.TimeoutExpired:
            print("  [timeout 1200s]", flush=True)
            failures.append((preset, "timeout"))

    print(f"\nDone. Failures: {len(failures)}")
    for preset, why in failures:
        print(f"  {preset}: {why}")


if __name__ == "__main__":
    main()