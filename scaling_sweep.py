#!/usr/bin/env python3
"""
scaling_sweep.py
================
End-to-end scaling sweep across body counts N ∈ {10, 25, 50, 100}.

For each N, the sweep now builds a per-model dataset because each
architecture has a different recommended data/epoch budget:

    MLP : 20 simulations, 100 epochs
    LSTM: 15 simulations,  80 epochs
    GNN : 10 simulations,  50 epochs

Directory layout per N:
    raw_data/N{n}/mlp/        raw trajectories sized for MLP
    raw_data/N{n}/lstm/       raw trajectories sized for LSTM
    raw_data/N{n}/gnn/        raw trajectories sized for GNN
    ml_ready_data/N{n}/mlp/   ML-ready .npz for MLP
    ml_ready_data/N{n}/lstm/  ML-ready .npz for LSTM
    ml_ready_data/N{n}/gnn/   ML-ready .npz for GNN
    training_runs/N{n}/mlp/   MLP checkpoint
    training_runs/N{n}/lstm/  LSTM checkpoint
    training_runs/N{n}/gnn/   GNN checkpoint
    results/N{n}/metrics.json combined evaluation

The export is model-agnostic, but each .npz is tagged with the model type
in its sidecar JSON for traceability.

Run on Google Colab (or locally). Designed to be uploaded alongside the
rest of the project; all paths resolve relative to this script's dir.

Usage (Colab)
------------
    !python scaling_sweep.py                  # full sweep
    !python scaling_sweep.py --quick          # 2 sims, 200 frames, 2 epochs

Caveats
-------
- Each N runs all 3 trainings back-to-back; budget ≈ N=10: ~25 min,
  N=25: ~40 min, N=50: ~90 min, N=100: ~3 h on a Colab T4.
- The script *resumes*: if a per-N output dir already exists with
  `model_best.pt`, that step is skipped. Delete the dir to force re-run.
- All heavy work (training, big simulations) happens here in Colab; this
  file is just orchestration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from utils import configure_utf8_stdout

configure_utf8_stdout()

from pipeline_config import (
    DEFAULT_N_FRAMES,
    GNN_EPOCHS,
    GNN_NUM_SIMULATIONS,
    HORIZON,
    IC_BASE_SEED,
    LSTM_EPOCHS,
    LSTM_NUM_SIMULATIONS,
    MLP_EPOCHS,
    MLP_NUM_SIMULATIONS,
    RESULTS_DIR,
    STRIDE,
    SWEEP_N_VALUES,
    SWEEP_ROLLOUT_K,
    TRAINING_RUNS_DIR,
    WINDOW_SIZE,
)


# ── Defaults ────────────────────────────────────────────────────────────
N_VALUES        = SWEEP_N_VALUES
FRAMES          = DEFAULT_N_FRAMES
WINDOW          = WINDOW_SIZE
ROLLOUT_K       = SWEEP_ROLLOUT_K
SEED            = IC_BASE_SEED
QUICK_NUM_SIMS  = 2
QUICK_FRAMES    = 200
QUICK_EPOCHS    = 2

MODELS = {
    "mlp":  {"num_sims": MLP_NUM_SIMULATIONS,  "epochs": MLP_EPOCHS},
    "lstm": {"num_sims": LSTM_NUM_SIMULATIONS, "epochs": LSTM_EPOCHS},
    "gnn":  {"num_sims": GNN_NUM_SIMULATIONS,  "epochs": GNN_EPOCHS},
}


# ── Pretty printing ────────────────────────────────────────────────────
def _hr(t: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n  {t}\n{bar}", flush=True)


def _step(label: str) -> None:
    print(f"\n>>> {label}", flush=True)


def run(cmd: list[str], cwd: str | Path | None = None) -> None:
    """Run a subprocess, tee stdout, raise on failure."""
    print("    $ " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"[fail] exit={e.returncode}  cmd={cmd}")


def exists_nonempty(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any(p.iterdir())


def _npz_path(ml_dir: Path) -> Path:
    """Return the unnormalised .npz path produced by 3d_export_pipeline.py."""
    return ml_dir / f"dataset_3d_w{WINDOW}h{HORIZON}s{STRIDE}r.npz"


# ── Per-N pipeline ─────────────────────────────────────────────────────
def run_one_N(n: int,
              frames: int,
              epochs: dict[str, int],
              num_sims: dict[str, int],
              test_frac: float,
              project_root: str | Path,
              dry_run: bool = False) -> dict:
    """Run the full per-model pipeline for one body count."""
    t_n = time.perf_counter()

    root = Path(project_root)
    train_dir = root / TRAINING_RUNS_DIR / f"N{n}"
    res_dir   = root / RESULTS_DIR       / f"N{n}"

    for d in (train_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_dirs = {m: train_dir / m for m in MODELS}
    ckpts      = {m: model_dirs[m] / "model_best.pt" for m in MODELS}
    raw_dirs   = {m: root / "raw_data"      / f"N{n}" / m for m in MODELS}
    ml_dirs    = {m: root / "ml_ready_data" / f"N{n}" / m for m in MODELS}
    npzs       = {m: _npz_path(ml_dirs[m]) for m in MODELS}

    metrics_path = res_dir / "metrics.json"

    # ── 1. generate_dataset per model ──────────────────────────────
    for m in MODELS:
        if not exists_nonempty(str(raw_dirs[m])):
            _step(f"N={n}  step 1/6  generate_dataset {m.upper()} "
                  f"({num_sims[m]} sims, {frames} frames)")
            cmd = [
                sys.executable, "simulation_3d.py",
                "--num-simulations", str(num_sims[m]),
                "--frames",          str(frames),
                "--base-seed",       str(SEED),
                "--N",               str(n),
                "--output-dir",      str(raw_dirs[m]),
                "--model-type",      m,
            ]
            if not dry_run:
                run(cmd, cwd=project_root)
        else:
            print(f"  [skip] {raw_dirs[m]} exists")

    # ── 2. 3d_export_pipeline.py per model ─────────────────────────
    # 3d_export_pipeline.py appends a normalisation suffix: 'r' for raw,
    # 'z' for z-scored. The sweep uses the default (raw / unnormalised).
    for m in MODELS:
        if not npzs[m].is_file():
            _step(f"N={n}  step 2/6  3d_export_pipeline {m.upper()} "
                  f"(W={WINDOW},H={HORIZON},S={STRIDE})")
            cmd = [
                sys.executable, "3d_export_pipeline.py",
                "--raw-dir",    str(raw_dirs[m]),
                "--export-dir", str(ml_dirs[m]),
                "--window",     str(WINDOW),
                "--horizon",    str(HORIZON),
                "--stride",     str(STRIDE),
                "--test-frac",  str(test_frac),
                "--model-type", m,
            ]
            if not dry_run:
                run(cmd, cwd=project_root)
        else:
            print(f"  [skip] {npzs[m]} exists")

    # ── 3–5. train each architecture on its own dataset ───────────
    # Large batches: on the 1-CPU GPU VM the per-batch overhead (single-thread
    # data load + CPU<->GPU .item() syncs) dominates, so the RTX 6000 sits at
    # ~0% util with batch=64 and epochs take many minutes. Bumping the batch
    # amortizes that overhead, far fewer iterations per epoch, negligible
    # VRAM cost (N<=100 tensors are tiny vs the 24 GB on the RTX 6000).
    trainers = {
        "mlp":  ("mlp_train.py", "512"),
        "lstm": ("lstm_train.py",  "256"),
        "gnn":  ("gnn_train.py",  "128"),
    }
    for idx, (m, (script, batch)) in enumerate(trainers.items(), start=3):
        if not ckpts[m].is_file():
            _step(f"N={n}  step {idx}/6  train {m.upper()}  "
                  f"({epochs[m]} epochs)")
            cmd = [
                sys.executable, script,
                "--npz",       str(npzs[m]),
                "--out",       str(model_dirs[m]),
                "--epochs",    str(epochs[m]),
                "--batch-size", batch,
            ]
            if not dry_run:
                run(cmd, cwd=project_root)
        else:
            print(f"  [skip] {ckpts[m]} exists")

    # ── 6. evaluate_models.py ─────────────────────────────────────
    if not metrics_path.is_file():
        _step(f"N={n}  step 6/6  evaluate (rollout-K={ROLLOUT_K})")
        # Evaluate on the MLP test set as the common hold-out. Because the
        # three datasets are generated from the same RNG family and split with
        # the same seed, their test sets are statistically comparable.
        cmd = [
            sys.executable, "evaluate_models.py",
            "--ckpt", f"{ckpts['mlp']}:mlp",
            "--ckpt", f"{ckpts['lstm']}:lstm",
            "--ckpt", f"{ckpts['gnn']}:gnn",
            "--npz",             str(npzs["mlp"]),
            "--split",           "test",
            "--rollout-K",       str(ROLLOUT_K),
            "--rollout-batches", "4",
            "--batch-size",      "32",
            "--json",            str(metrics_path),
        ]
        if not dry_run:
            run(cmd, cwd=project_root)
    else:
        print(f"  [skip] {metrics_path} exists")

    elapsed = time.perf_counter() - t_n
    print(f"\n  ✓ N={n} done in {elapsed/60:.1f} min", flush=True)

    return {"N": n, "elapsed_sec": elapsed, "metrics": str(metrics_path)}


# ── Sweep entry point ──────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-end N-body scaling sweep with per-model defaults."
    )
    p.add_argument("--N", type=int, nargs="+", default=N_VALUES,
                   help=f"Body counts to sweep (default: {N_VALUES})")
    p.add_argument("--num-simulations", type=int, default=None,
                   help="Override sim count for ALL models (default: per-model)")
    p.add_argument("--mlp-num-simulations", type=int, default=MLP_NUM_SIMULATIONS,
                   help=f"MLP sims (default: {MLP_NUM_SIMULATIONS})")
    p.add_argument("--lstm-num-simulations", type=int, default=LSTM_NUM_SIMULATIONS,
                   help=f"LSTM sims (default: {LSTM_NUM_SIMULATIONS})")
    p.add_argument("--gnn-num-simulations", type=int, default=GNN_NUM_SIMULATIONS,
                   help=f"GNN sims (default: {GNN_NUM_SIMULATIONS})")
    p.add_argument("--frames", type=int, default=FRAMES,
                   help=f"Frames per sim (default: {FRAMES})")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs for ALL models (default: per-model)")
    p.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS,
                   help=f"MLP epochs (default: {MLP_EPOCHS})")
    p.add_argument("--lstm-epochs", type=int, default=LSTM_EPOCHS,
                   help=f"LSTM epochs (default: {LSTM_EPOCHS})")
    p.add_argument("--gnn-epochs", type=int, default=GNN_EPOCHS,
                   help=f"GNN epochs (default: {GNN_EPOCHS})")
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="Fraction of windows held out for final evaluation.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 sims, 200 frames, 2 epochs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent

    num_sims = {
        "mlp":  args.num_simulations if args.num_simulations is not None else args.mlp_num_simulations,
        "lstm": args.num_simulations if args.num_simulations is not None else args.lstm_num_simulations,
        "gnn":  args.num_simulations if args.num_simulations is not None else args.gnn_num_simulations,
    }
    epochs = {
        "mlp":  args.epochs if args.epochs is not None else args.mlp_epochs,
        "lstm": args.epochs if args.epochs is not None else args.lstm_epochs,
        "gnn":  args.epochs if args.epochs is not None else args.gnn_epochs,
    }

    frames    = args.frames
    test_frac = args.test_frac
    if args.quick:
        for m in MODELS:
            num_sims[m] = QUICK_NUM_SIMS
        frames = QUICK_FRAMES
        for m in MODELS:
            epochs[m] = QUICK_EPOCHS

    cfg = {
        "N": args.N,
        "frames": frames,
        "test_frac": test_frac,
        "num_simulations": num_sims,
        "epochs": epochs,
    }
    _hr(f"SCALING SWEEP  ·  N ∈ {args.N}  ·  frames={frames}  "
        f"·  test_frac={test_frac}\n"
        f"  MLP : sims={num_sims['mlp']},  epochs={epochs['mlp']}\n"
        f"  LSTM: sims={num_sims['lstm']}, epochs={epochs['lstm']}\n"
        f"  GNN : sims={num_sims['gnn']},  epochs={epochs['gnn']}")

    summary = {"config": cfg, "per_N": []}
    t0 = time.perf_counter()
    for n in args.N:
        _hr(f"N = {n}")
        rec = run_one_N(n, frames, epochs, num_sims, test_frac,
                        project_root, args.dry_run)
        summary["per_N"].append(rec)
    summary["total_sec"] = time.perf_counter() - t0

    log_path = project_root / RESULTS_DIR / "sweep_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ sweep summary → {log_path}")
    print(f"  total {summary['total_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
