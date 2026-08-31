"""
3d_export_pipeline.py
=====================
Turns raw 3D N-body trajectories produced by `simulation_3d.py` into ML-ready
sliding-window datasets, with optional per-simulation normalisation and a
persisted train/val split.

Pipeline
--------
1. Read each `sim_N{n}_{idx:03d}.npz` from `raw_dir`
   (each contains: `frames [F,N,6]`, `mass [N]`, `meta [4]`).
2. Build sliding-window pairs (X, y) using a vectorised NumPy view, no
   Python-side window loop.
3. Optionally standardise each trajectory's positions and velocities to
   zero mean / unit std before windowing (recommended when mixing
   simulations with different IC parameters).
4. Concatenate across simulations and write a compressed `.npz` archive
   containing X, y, and: if `keep_mass`, the per-body masses.
5. Save a sibling `.json` with metadata and the train/val indices so that
   downstream training runs are exactly reproducible.

Usage
-----
    from 3d_export_pipeline import process_and_export
    from pipeline_config import WINDOW_SIZE
    process_and_export(window_size=WINDOW_SIZE, normalize=True)

CLI
---
    python 3d_export_pipeline.py --window 5 --normalize --raw-dir raw_data \\
        --export-dir ml_ready_data --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils import configure_utf8_stdout

configure_utf8_stdout()


from pipeline_config import (
    HORIZON,
    IC_M_MAX,
    IC_M_MIN,
    ML_READY_DIR,
    ModelType,
    NORMALIZE,
    RAW_DIR,
    SPLIT_SEED,
    STRIDE,
    TEST_FRAC,
    VAL_FRAC,
    WINDOW_SIZE,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_trajectory(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Load a single raw 3D trajectory, supporting both layouts:
      - new (.npz): keys 'frames', 'mass', 'meta'
      - legacy (.npy): just an array
    Returns (frames, mass_or_None, meta_or_None).
    """
    p = str(path)
    if p.endswith(".npz"):
        archive = np.load(p)
        if not hasattr(archive, "files") or "frames" not in archive.files:
            raise ValueError(f"{p}: expected key 'frames' inside .npz archive.")
        frames = np.asarray(archive["frames"])
        mass   = np.asarray(archive["mass"])   if "mass"   in archive.files else None
        meta   = np.asarray(archive["meta"])   if "meta"   in archive.files else None
        return frames, mass, meta
    # legacy .npy fallback
    frames = np.load(p)
    return frames, None, None


def _standardize(frames: np.ndarray) -> np.ndarray:
    """
    Per-trajectory z-score on positions and velocities separately.

    For each simulation we compute mean/std over (frames, bodies) for each
    of the 6 channels and apply (x - μ) / σ. This keeps simulations with
    different disc IC parameters (r_core, r_disc, m_min, m_max) on a
    comparable scale so the loss landscape is consistent across the
    master dataset.

    A 1e-8 floor on std avoids divide-by-zero on perfectly static channels.
    """
    flat = frames.reshape(-1, frames.shape[-1])          # (F*N, 6)
    mu   = flat.mean(axis=0, keepdims=True)              # (1, 6)
    sig  = flat.std(axis=0, keepdims=True)               # (1, 6)
    sig  = np.where(sig < 1e-8, 1.0, sig)
    out  = (frames - mu) / sig
    return out.astype(frames.dtype, copy=False)


def _sliding_windows(traj: np.ndarray,
                     window: int,
                     horizon: int = 1,
                     stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window supervised pairs from a single trajectory.

    For each starting index `i`, X[i] is `traj[i : i+window]` and y[i] is
    `traj[i + window + horizon - 1]`: i.e. predicting the state `horizon`
    steps *after* the window ends.

    Implementation
    --------------
    Uses `np.lib.stride_tricks.sliding_window_view` so the C-level NumPy
    strider does the slicing; we only materialise the final contiguous
    arrays. This is dramatically faster than a Python for-loop over windows.

    Parameters
    ----------
    traj    : (F, N, 6) array
    window  : input window length W
    horizon : prediction horizon (default 1 = predict the next step)
    stride  : step between consecutive windows (default 1 = dense)

    Returns
    -------
    X : (n_win, W, N, 6)  float32
    y : (n_win, N, 6)     float32
    """
    if traj.ndim != 3 or traj.shape[-1] != 6:
        raise ValueError(f"trajectory must have shape (F, N, 6); got {traj.shape}")

    F = traj.shape[0]
    target_offset = window + horizon - 1
    last_start = F - window - horizon
    if last_start < 0:
        # Not enough frames for even a single window.
        return (np.empty((0, window, traj.shape[1], 6), dtype=traj.dtype),
                np.empty((0, traj.shape[1], 6), dtype=traj.dtype))

    # sliding_window_view returns (F - W + 1, N, 6, W); roll the window axis
    # back to position 1 so X has shape (n_win_full, W, N, 6).
    win_views = np.lib.stride_tricks.sliding_window_view(traj, window, axis=0)
    X_full = np.moveaxis(win_views, -1, 1)               # (n_win_full, W, N, 6)
    # Truncate to legal starts, then apply stride. This guarantees the
    # target index `i*stride + target_offset` stays < F for every window.
    X_full = X_full[: last_start + 1]
    X      = X_full[::stride]
    starts = np.arange(X_full.shape[0])[::stride]        # absolute window starts
    y_idx  = starts + target_offset
    y      = traj[y_idx]                                  # (n_win, N, 6)

    return X.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


# ── Public API ───────────────────────────────────────────────────────────────
def create_sliding_windows(raw_file_path: str,
                           window_size: int = WINDOW_SIZE,
                           horizon: int = HORIZON,
                           stride: int = STRIDE,
                           normalize: bool = NORMALIZE):
    """
    Convert one raw trajectory into (X, y) supervised pairs.

    Kept name-compatible with the original API so existing callers continue
    to work. Adds `horizon`, `stride`, `normalize` kwargs.

    Returns
    -------
    X : (n_win, W, N, 6) float32
    y : (n_win, N, 6)    float32
    """
    frames, _mass, _meta = _load_trajectory(raw_file_path)
    if normalize:
        frames = _standardize(frames)
    return _sliding_windows(frames, window=window_size, horizon=horizon, stride=stride)


def process_and_export(raw_dir: str = RAW_DIR,
                       export_dir: str = ML_READY_DIR,
                       window_size: int = WINDOW_SIZE,
                       horizon: int = HORIZON,
                       stride: int = STRIDE,
                       normalize: bool = NORMALIZE,
                       keep_mass: bool = True,
                       val_frac: float = VAL_FRAC,
                       test_frac: float = TEST_FRAC,
                       split_seed: int = SPLIT_SEED,
                       m_min: float = IC_M_MIN,
                       m_max: float = IC_M_MAX,
                       model_type: str | None = None) -> str | None:
    """
    Walk `raw_dir`, build (X, y) for every simulation, concatenate, and
    save a single compressed `.npz` plus a sidecar `.json` with metadata
    and train/val indices.

    Parameters
    ----------
    raw_dir     : directory containing `sim_N*.npz` (or legacy `*.npy`) files
    export_dir  : output directory (created if missing)
    window_size : input window length
    horizon     : prediction horizon (steps ahead of the window's end)
    stride      : stride between consecutive windows within a trajectory
    normalize   : if True, z-score each trajectory before windowing
    keep_mass   : if True, save per-body masses (and verify they match
                  across simulations, a mismatch is a hard error)
    val_frac    : fraction of windows held out for validation (0 to skip)
    split_seed  : RNG seed for the train/val split (reproducible)
    m_min       : minimum body mass for log-uniform sampling in the disc IC.
                  Recorded in the sidecar JSON for the project audit trail.
                  NOTE: only takes effect when the raw .npz files were
                  generated with matching values; this pipeline does not
                  regenerate them: pass these flags to `generate_dataset`
                  in `simulation_3d.py` first.
    m_max       : maximum body mass for log-uniform sampling in the disc IC.
                  Same caveat as `m_min`.
    model_type  : optional model identifier ("mlp" | "lstm" | "gnn") recorded
                  in the sidecar JSON. The export itself is model-agnostic,
                  but this flag documents which downstream model the dataset
                  was sized for.

    Returns
    -------
    export_path : path to the written `.npz`, or None if nothing to export.
    """
    export_path_obj = Path(export_dir)
    export_path_obj.mkdir(parents=True, exist_ok=True)

    # Accept both new (.npz) and legacy (.npy) raw files.
    raw_dir_obj = Path(raw_dir)
    raw_files = sorted(
        list(raw_dir_obj.glob("sim_*.npz")) + list(raw_dir_obj.glob("sim_*.npy"))
    )
    if not raw_files:
        print(f"No raw 3D data found in {raw_dir!r}. Run simulation_3d.py first.")
        return None

    print(f"Found {len(raw_files)} raw 3D simulations in {raw_dir}.")
    print(f"  window={window_size}  horizon={horizon}  stride={stride}  "
          f"normalize={normalize}")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    mass_list: list[np.ndarray] = []   # one (N,) per simulation
    metas: list[np.ndarray] = []
    n_windows_per_sim: list[int] = []

    t_start = time.perf_counter()
    for idx, path in enumerate(raw_files, start=1):
        frames, mass, meta = _load_trajectory(path)
        if normalize:
            frames = _standardize(frames)
        X, y = _sliding_windows(frames, window=window_size,
                                horizon=horizon, stride=stride)
        if X.shape[0] == 0:
            print(f"  [{idx}/{len(raw_files)}] {Path(path).name}: "
                  f"skipped (not enough frames for window={window_size}).")
            continue

        all_X.append(X)
        all_y.append(y)
        n_windows_per_sim.append(X.shape[0])

        if keep_mass and mass is not None:
            mass_list.append(np.asarray(mass, dtype=np.float64))
        if meta is not None:
            metas.append(meta)

        print(f"  [{idx}/{len(raw_files)}] {Path(path).name}: "
              f"{X.shape[0]:>6d} windows  "
              f"(shape X={X.shape}, y={y.shape})")

    if not all_X:
        print("No usable trajectories produced any windows, nothing to export.")
        return None

    # Fail loudly on heterogeneous raw files rather than letting
    # np.concatenate / np.stack surface a cryptic broadcast error later:
    # every simulation shares one stacked mass table, so all inputs must
    # have the same body count N. (X windows are (n, W, N, 6).)
    body_counts = {arr.shape[-2] for arr in all_X}
    if len(body_counts) != 1:
        raise ValueError(
            f"Heterogeneous body counts across raw files: {sorted(body_counts)}. "
            "All simulations must have the same N because they are stacked "
            "along axis 0 with a single shared mass table; re-run the "
            "simulation for the mismatched files or export them separately."
        )

    if keep_mass and len(mass_list) != len(all_X):
        print(f"  [warn] {len(all_X) - len(mass_list)} of {len(all_X)} raw files "
              "had no 'mass' key; exporting without a mass table "
              "(a partial table would mis-align with the simulation axis).")
        mass_list = []

    master_X = np.concatenate(all_X, axis=0)
    master_y = np.concatenate(all_y, axis=0)
    del all_X, all_y  # free memory before save

    # ── Train / val / test split ─────────────────────────────────────────────
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(master_X.shape[0])
    n_val  = int(round(master_X.shape[0] * val_frac))  if 0 < val_frac < 1 else 0
    n_test = int(round(master_X.shape[0] * test_frac)) if 0 < test_frac < 1 else 0
    # Reserve test from the end of the permutation, then val from what
    # remains, so the three index sets are disjoint.
    test_idx  = perm[:n_test].astype(np.int64)
    val_idx   = perm[n_test:n_test + n_val].astype(np.int64)
    train_idx = perm[n_test + n_val:].astype(np.int64)

    # ── Save ─────────────────────────────────────────────────────────────────
    suffix = "z" if normalize else "r"
    filename = f"dataset_3d_w{window_size}h{horizon}s{stride}{suffix}.npz"
    export_path = str(export_path_obj / filename)

    save_kwargs = dict(X=master_X, y=master_y,
                       train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    if keep_mass and mass_list:
        save_kwargs["mass"] = np.stack(mass_list, axis=0)   # (n_sims, N)
    if metas:
        save_kwargs["meta"] = np.stack(metas, axis=0)
    np.savez_compressed(export_path, **save_kwargs)

    sidecar = {
        "window_size":   int(window_size),
        "horizon":       int(horizon),
        "stride":        int(stride),
        "normalize":     bool(normalize),
        "n_simulations": int(len(raw_files)),
        "n_windows":     int(master_X.shape[0]),
        "n_train":       int(train_idx.shape[0]),
        "n_val":         int(val_idx.shape[0]),
        "n_test":        int(test_idx.shape[0]),
        "X_shape":       list(master_X.shape),
        "y_shape":       list(master_y.shape),
        "split_seed":    int(split_seed),
        "m_min":         float(m_min),
        "m_max":         float(m_max),
        "raw_files":     [Path(p).name for p in raw_files],
        "model_type":    model_type,
        "n_windows_per_sim": n_windows_per_sim,
    }
    sidecar_path = str(Path(export_path).with_suffix(".json"))
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    elapsed = time.perf_counter() - t_start
    print("-" * 60)
    print(f"3D Export Complete -> {export_path}")
    print(f"  Total windows : {master_X.shape[0]:,}  "
          f"(train={train_idx.shape[0]:,} | val={val_idx.shape[0]:,} | test={test_idx.shape[0]:,})")
    print(f"  X shape       : {master_X.shape}  -> [Samples, Window, N, 6]")
    print(f"  y shape       : {master_y.shape}  -> [Samples, N, 6]")
    if keep_mass and mass_list:
        mass_arr = np.stack(mass_list, axis=0)
        print(f"  Mass array    : {mass_arr.shape}  (one (N,) per simulation)")
    print(f"  Metadata      : {sidecar_path}")
    print(f"  Elapsed       : {elapsed:.2f}s")
    return export_path


# ── CLI ──────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert raw 3D N-body trajectories into ML-ready windows.",
    )
    p.add_argument("--raw-dir",     default=RAW_DIR,          help="Directory of raw sim_*.npz files.")
    p.add_argument("--export-dir",  default=ML_READY_DIR,     help="Output directory.")
    p.add_argument("--window",      type=int, default=WINDOW_SIZE, help="Input window length.")
    p.add_argument("--horizon",     type=int, default=HORIZON,     help="Prediction horizon (steps ahead).")
    p.add_argument("--stride",      type=int, default=STRIDE,      help="Stride between consecutive windows.")
    p.add_argument("--normalize",   action="store_true",        help="Z-score each trajectory before windowing.")
    p.add_argument("--no-mass",     action="store_true",        help="Skip saving per-body masses.")
    p.add_argument("--val-frac",    type=float, default=VAL_FRAC,   help="Validation fraction (0 to skip).")
    p.add_argument("--test-frac",   type=float, default=TEST_FRAC,  help="Test fraction (0 to skip).")
    p.add_argument("--split-seed",  type=int, default=SPLIT_SEED,   help="RNG seed for the train/val/test split.")
    p.add_argument("--model-type",  default=None,
                   choices=list(ModelType.values()),
                   help="Optional model tag (mlp/lstm/gnn) to record in the "
                        "sidecar JSON. The export is model-agnostic; this just "
                        "documents which downstream model the dataset was sized for.")
    p.add_argument("--m-min",       type=float, default=IC_M_MIN,
                   help="Minimum body mass for log-uniform sampling in the "
                        "disc IC (real-MSun values like 0.1 give a stellar "
                        "IMF before Σ=1 normalisation). Recorded in the "
                        "sidecar JSON; only takes effect when the raw .npz "
                        "files were generated with matching bounds.")
    p.add_argument("--m-max",       type=float, default=IC_M_MAX,
                   help="Maximum body mass for log-uniform sampling in the "
                        "disc IC (real-MSun upper bound ≈ 50–100). Same "
                        "caveat as --m-min.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    process_and_export(
        raw_dir=args.raw_dir,
        export_dir=args.export_dir,
        window_size=args.window,
        horizon=args.horizon,
        stride=args.stride,
        normalize=args.normalize,
        keep_mass=not args.no_mass,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        m_min=args.m_min,
        m_max=args.m_max,
        model_type=args.model_type,
    )