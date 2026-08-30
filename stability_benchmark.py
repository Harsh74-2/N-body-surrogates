#!/usr/bin/env python3
"""
stability_benchmark.py
======================
Rollout-stability benchmark for the trained N-body surrogates.

Where `evaluate_models.py` collapses long-horizon behaviour into a single
scalar (mean K-step rollout energy drift), this script records the **full
per-step curve** over a K=128-step autoregressive rollout against the *true*
trajectory and fits its **gradient (slope vs step)**, the "rollout stability
gradient", "energy gradient", and "loss gradient" the project reports.

For every checkpoint it records, for k = 1..K:
    mse(k)              : positional+velocity MSE vs the true frame at step k
    pos_mse(k), vel_mse(k), position-only / velocity-only MSE
    energy_drift(k)     , |E(pred_k) − E0| / |E0|  (physics consistency)
    energy_err_vs_true(k)- |E(pred_k) − E(true_k)| / |E(true_k)|
    loss(k)             , mse(k) + w_energy * energy_drift(k)
and the linear-fit slope (+ R²) of each curve vs k, a log-slope for
characterising exponential blow-up, and the first divergence step.

Methodology (deliberate, documented)
------------------------------------
Every surrogate (MLP, LSTM, GNN) is rolled out with a **sliding window**:
seed with the true W=5 window ending at the start frame, predict the next
state, then shift the window with the prediction and repeat. This is
*in-distribution* -- the models were trained on real W-windows -- so the
long-horizon error is genuine stability error, not an out-of-distribution
artefact of a degenerate window. The sliding window is essential for the
GNN: its `forward` iterates over the W timesteps (gnn_train.py:236-260) and
only runs message passing for ``t in range(1, W)``, so a 1-frame
`model.step` window (W=1) runs *zero* message-passing rounds -- an
out-of-distribution path that inflated GNN mse@1 ~30x and energy ~140x in
an earlier version. The same sliding-window path is now used by the OOD
runner (`real_case_runner._rollout_sliding`), the rollout-energy training
loss (`losses.rollout_energy_loss`), and the sweep eval
(`evaluate_models.py`), so training, sweep eval, this stability benchmark,
and the OOD validation all measure one consistent in-distribution path.

The chosen method is stored in each JSON record under `rollout_method`.

Reuse
-----
- `evaluate_models.build_model` , load a checkpoint + build the right
  `*Surrogate` class from its saved config.
- `losses.total_energy` (re-exported by evaluate_models), KE + softened PE,
  identical to the formula the engine and training losses use.
- The sliding-window shift mirrors `real_case_validation.real_case_runner.
  _rollout_sliding` (which all three models now use). It is re-implemented
  inline (≈10 lines) so this script does not pull the real-case solar-system
  import chain.

Usage
-----
    # One N, several checkpoints (N=25 gets 3 single-step + 3 stable)
    python stability_benchmark.py \\
        --ckpt training_runs/N25/mlp/model_best.pt:mlp \\
        --ckpt training_runs/N25/lstm/model_best.pt:lstm \\
        --ckpt training_runs/N25/gnn/model_best.pt:gnn \\
        --N 25 --K 128 --rollout-batches 8 \\
        --json results/N25/stability.json --out plots

    # Aggregate every results/N*/stability.json into a summary + overview plot
    python stability_benchmark.py --aggregate-only --out plots
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# matplotlib Agg backend, no display server needed on the VM.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import configure_utf8_stdout, load_sibling_module, pick_device  # noqa: E402

configure_utf8_stdout()

from pipeline_config import (  # noqa: E402
    DEFAULT_EPS,
    DEFAULT_GRAVITY_G,
    FEATURE_DIM,
    WINDOW_SIZE,
)

# Reuse evaluate_models.build_model (loads ckpt + builds the *Surrogate class)
# and the shared total_energy helper. evaluate_models already loads losses.py
# and the three trainer modules, so this single import gives us everything.
_eval_mod = load_sibling_module("nbody_eval", "evaluate_models.py")
build_model = _eval_mod.build_model
total_energy = _eval_mod.total_energy


# ── Raw trajectory loading ───────────────────────────────────────────────────
# The raw trajectories live in one of two layouts:
#   (a) raw_data/sim_N{N}_XXX.npz             , legacy (e.g. N25 root files)
#   (b) raw_data/N{N}/{mlp,lstm,gnn}/sim_...  , sweep layout (N10/50/100)
# For a given (N, sim_idx) the trajectory is identical across model subdirs
# (ICs are seeded by base_seed + N, not model type), so any one subdir is a
# valid ground-truth source. We prefer the root layout, then the `mlp` subdir.

def resolve_raw_root(raw_dir: Path, N: int, subdir: str | None = None) -> Path:
    """Return the directory that actually holds sim_N{N}_*.npz for this N."""
    root = Path(raw_dir)
    if subdir:
        cands = [root / f"N{N}" / subdir, root]
    else:
        cands = [root, root / f"N{N}" / "mlp", root / f"N{N}" / "lstm",
                 root / f"N{N}" / "gnn", root / f"N{N}"]
    for c in cands:
        if c.is_dir() and any(c.glob(f"sim_N{N}_*.npz")):
            return c
    raise FileNotFoundError(
        f"no raw_data/sim_N{N}_*.npz found under {root} "
        f"(checked root and N{N}/{{mlp,lstm,gnn}}/).")


def load_raw_sim(raw_root: Path, N: int, sim_idx: int) -> dict:
    """Load sim_N{N}_{idx:03d}.npz from `raw_root` → frames (F,N,6), mass (N,)."""
    path = raw_root / f"sim_N{N}_{sim_idx:03d}.npz"
    if not path.is_file():
        alt = raw_root / f"sim_N{N}_{sim_idx}.npz"   # non-zero-padded fallback
        path = alt if alt.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"raw sim not found: {path}")
    d = np.load(path, allow_pickle=False)
    frames = np.ascontiguousarray(d["frames"], dtype=np.float32)   # (F, N, 6)
    mass = np.ascontiguousarray(d["mass"], dtype=np.float32)       # (N,)
    return {"frames": frames, "mass": mass, "path": str(path)}


def available_sims(raw_root: Path, N: int) -> list[int]:
    """Return sorted sim indices present under `raw_root` for this N."""
    idxs = []
    for p in raw_root.glob(f"sim_N{N}_*.npz"):
        stem = p.stem.replace(f"sim_N{N}_", "")
        try:
            idxs.append(int(stem))
        except ValueError:
            continue
    return sorted(idxs)


def select_start_frames(raw_root: Path, N: int, K: int, W: int,
                        n_starts: int, stride: int, n_sims: int
                        ) -> list[tuple[int, int]]:
    """
    Pick (sim_idx, frame_idx) start points spread over the first `n_sims`
    simulations, spaced `stride` frames apart, each leaving K+W future frames.

    Requires frame >= W-1 so a full true W-window can seed the rollout.
    """
    sims = available_sims(raw_root, N)
    if not sims:
        raise FileNotFoundError(
            f"no raw_data/sim_N{N}_*.npz found under {raw_root}")
    if len(sims) < n_sims:
        print(f"  [warn] only {len(sims)} sim(s) available for N={N}; "
              f"using all of them (requested {n_sims}).")
    sims = sims[:n_sims]

    starts: list[tuple[int, int]] = []
    per_sim = max(1, math.ceil(n_starts / len(sims)))
    for sim_idx in sims:
        frames = load_raw_sim(raw_root, N, sim_idx)["frames"]
        F = frames.shape[0]
        # Need frame in [W-1, F-K-1] so the W-window and K future frames exist.
        lo, hi = W - 1, F - K - 1
        if hi <= lo:
            print(f"  [warn] sim {sim_idx} too short (F={F}) for K={K}, W={W}; "
                  f"skipping.")
            continue
        candidates = list(range(lo, hi + 1, stride))
        if not candidates:
            candidates = [lo]
        # Spread starts: take evenly across candidates.
        if len(candidates) > per_sim:
            sel = [candidates[int(i * (len(candidates) - 1) / max(per_sim - 1, 1))]
                   for i in range(per_sim)]
        else:
            sel = candidates
        for f in sel:
            starts.append((sim_idx, int(f)))
        if len(starts) >= n_starts:
            break
    starts = starts[:n_starts]
    if len(starts) < n_starts:
        print(f"  [warn] could only collect {len(starts)}/{n_starts} start "
              f"frames for N={N}.")
    return starts


# ── Autoregressive rollouts (in-distribution) ────────────────────────────────
def rollout_sliding(model: torch.nn.Module,
                    window0: torch.Tensor,    # (W, N, F)
                    mass_t: torch.Tensor,     # (1, N)
                    n_steps: int) -> np.ndarray:
    """
    Sliding-window autoregressive rollout for MLP / LSTM.

    Seeds with the true W-window, predicts the next state, shifts the window
    with the prediction, and repeats. Returns (n_steps+1, N, F) numpy;
    row 0 is the warm-up anchor (the last true frame of `window0`).
    """
    states = [window0[-1]]
    state = window0.unsqueeze(0)                          # (1, W, N, F)
    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(state, mass_t)                  # (1, N, F)
            states.append(pred[0])
            state = torch.cat([state[:, 1:], pred.unsqueeze(1)], dim=1)
    return torch.stack(states, dim=0).cpu().numpy()       # (n_steps+1, N, F)


def run_rollout(model: torch.nn.Module, model_type: str,
                window0: torch.Tensor, mass_t: torch.Tensor,
                K: int) -> np.ndarray:
    """In-distribution sliding-window rollout for every model.

    All three surrogates were trained on W=5 windows (dataset_3d_w5h1s1r),
    so every model is rolled out by seeding with the true W-window and sliding
    it with each prediction. The GNN's `forward` iterates over the W timesteps
    (gnn_train.py:236-260) and accumulates temporal context, so a 1-frame
    `model.step` window -- used by an earlier version -- is out-of-distribution
    for a model trained on W=5 and inflated GNN mse@1 ~30x / energy ~140x over
    the in-distribution sweep eval. Use the W-window for all models.
    """
    if model_type in ("mlp", "lstm", "gnn"):
        return rollout_sliding(model, window0, mass_t, K)
    raise ValueError(f"unknown model_type: {model_type}")


# ── Per-step metrics ─────────────────────────────────────────────────────────
def per_step_mse(pred: np.ndarray, true: np.ndarray) -> dict:
    """pred, true: (K, N, 6) → {mse, pos_mse, vel_mse} each (K,)."""
    diff_sq = (pred - true) ** 2
    mse = diff_sq.mean(axis=(1, 2))                       # over N and F
    pos_mse = diff_sq[:, :, :3].mean(axis=(1, 2))
    vel_mse = diff_sq[:, :, 3:6].mean(axis=(1, 2))
    return {"mse": mse, "pos_mse": pos_mse, "vel_mse": vel_mse}


def _energy_series(states: np.ndarray, mass_t: torch.Tensor,
                   eps: float, g: float) -> np.ndarray:
    """E(k) for a (T, N, 6) trajectory via losses.total_energy (torch, no grad)."""
    T = states.shape[0]
    E = np.empty(T, dtype=np.float64)
    with torch.no_grad():
        for k in range(T):
            s = torch.as_tensor(states[k], dtype=torch.float32, device=mass_t.device)
            E[k] = float(total_energy(s[..., :3], s[..., 3:6], mass_t,
                                       eps=eps, g=g).item())
    return E


def per_step_energy(pred_steps: np.ndarray, true_steps: np.ndarray,
                    anchor: np.ndarray, mass_t: torch.Tensor,
                    eps: float, g: float, eps_floor: float = 1e-8) -> dict:
    """
    pred_steps, true_steps: (K, N, 6): the K predicted / true frames.
    anchor: (N, 6): the true start frame (= pred warm-up anchor), E0 source.

    Returns {energy_drift, energy_err_vs_true} each (K,).
      energy_drift      = |E(pred_k) − E0| / |E0|        (physics consistency)
      energy_err_vs_true= |E(pred_k) − E(true_k)| / |E(true_k)|
    """
    E_pred = _energy_series(pred_steps, mass_t, eps, g)
    E_true = _energy_series(true_steps, mass_t, eps, g)
    E0 = float(_energy_series(anchor[None], mass_t, eps, g)[0])

    denom0 = max(abs(E0), eps_floor)
    energy_drift = np.abs(E_pred - E0) / denom0
    denom_t = np.maximum(np.abs(E_true), eps_floor)
    energy_err_vs_true = np.abs(E_pred - E_true) / denom_t
    return {"energy_drift": energy_drift, "energy_err_vs_true": energy_err_vs_true}


def composed_loss(mse: np.ndarray, energy_drift: np.ndarray,
                  w_energy: float) -> np.ndarray:
    """loss(k) = mse(k) + w_energy * energy_drift(k)."""
    return mse + w_energy * energy_drift


def find_divergence(pred_steps: np.ndarray, mse: np.ndarray,
                    blowup_thresh: float) -> int | None:
    """First 1-indexed step k where pred is non-finite or mse[k] > threshold."""
    if not np.isfinite(pred_steps).all():
        # First non-finite frame (0-indexed) → step k = idx+1.
        bad = np.where(~np.isfinite(pred_steps).all(axis=(1, 2)))[0]
        if bad.size:
            return int(bad[0]) + 1
    over = np.where(mse > blowup_thresh)[0]
    if over.size:
        return int(over[0]) + 1
    return None


# ── Slope fits ───────────────────────────────────────────────────────────────
def fit_slopes(metric: np.ndarray, k: np.ndarray
               ) -> tuple[float, float]:
    """Linear fit (degree 1) of metric vs k → (slope, R²). NaN-aware."""
    mask = np.isfinite(metric)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    coeffs = np.polyfit(k[mask], metric[mask], 1)
    slope = float(coeffs[0])
    pred_lin = np.polyval(coeffs, k[mask])
    ss_res = float(np.sum((metric[mask] - pred_lin) ** 2))
    ss_tot = float(np.sum((metric[mask] - metric[mask].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, float(r2)


def fit_log_slopes(metric: np.ndarray, k: np.ndarray,
                   eps_floor: float = 1e-12) -> float:
    """Slope of log(metric + eps_floor) vs k: characterises exponential
    blow-up (most informative for diverging GNN). NaN if any non-positive."""
    mask = np.isfinite(metric) & (metric > 0)
    if mask.sum() < 2:
        return float("nan")
    coeffs = np.polyfit(k[mask], np.log(metric[mask] + eps_floor), 1)
    return float(coeffs[0])


# ── Per-model benchmark ──────────────────────────────────────────────────────
def run_one_model(ckpt_path: str, model_type: str, device: torch.device,
                  raw_root: Path, N: int, K: int,
                  starts: list[tuple[int, int]],
                  eps: float, g: float, w_energy: float,
                  blowup_thresh: float) -> dict:
    """Run K-step rollouts from every start frame; reduce to mean curves + slopes."""
    print(f"\n[rollout] {model_type.upper()}  ckpt={ckpt_path}")
    model = build_model(ckpt_path, model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    W = getattr(model, "window_size", WINDOW_SIZE)
    rollout_method = "sliding_window"   # all models trained on W=5 windows
    print(f"  {n_params:,} params  W={W}  method={rollout_method}  "
          f"K={K}  starts={len(starts)}")

    # Per-start curve stacks.
    mse_all = np.full((len(starts), K), np.nan, dtype=np.float64)
    pos_all = np.full((len(starts), K), np.nan, dtype=np.float64)
    vel_all = np.full((len(starts), K), np.nan, dtype=np.float64)
    edr_all = np.full((len(starts), K), np.nan, dtype=np.float64)
    eet_all = np.full((len(starts), K), np.nan, dtype=np.float64)
    div_steps: list[int] = []

    for i, (sim_idx, frame) in enumerate(starts):
        sim = load_raw_sim(raw_root, N, sim_idx)
        frames = sim["frames"]
        mass = sim["mass"]
        mass_t = torch.as_tensor(mass, dtype=torch.float32,
                                 device=device).unsqueeze(0)         # (1, N)

        window0 = torch.as_tensor(frames[frame - W + 1:frame + 1],
                                  dtype=torch.float32, device=device)  # (W, N, 6)
        anchor = frames[frame]                                         # (N, 6)
        true_steps = frames[frame + 1:frame + 1 + K]                   # (K, N, 6)

        traj = run_rollout(model, model_type, window0, mass_t, K)      # (K+1, N, 6)
        pred_steps = traj[1:K + 1]                                     # (K, N, 6)

        if pred_steps.shape != true_steps.shape:
            print(f"  [warn] start {i} (sim {sim_idx}, frame {frame}): "
                  f"shape mismatch {pred_steps.shape} vs {true_steps.shape}; skipping.")
            continue

        m = per_step_mse(pred_steps, true_steps)
        e = per_step_energy(pred_steps, true_steps, anchor, mass_t, eps, g)

        div = find_divergence(pred_steps, m["mse"], blowup_thresh)
        if div is not None:
            # Truncate this start's curves from the divergence step onward.
            m["mse"][div - 1:] = np.nan
            m["pos_mse"][div - 1:] = np.nan
            m["vel_mse"][div - 1:] = np.nan
            e["energy_drift"][div - 1:] = np.nan
            e["energy_err_vs_true"][div - 1:] = np.nan
            div_steps.append(div)
        else:
            div_steps.append(K + 1)  # sentinel: no divergence within horizon

        mse_all[i] = m["mse"]
        pos_all[i] = m["pos_mse"]
        vel_all[i] = m["vel_mse"]
        edr_all[i] = e["energy_drift"]
        eet_all[i] = e["energy_err_vs_true"]

    # Aggregate across starts (NaN-aware).
    mse_mean = np.nanmean(mse_all, axis=0)
    pos_mean = np.nanmean(pos_all, axis=0)
    vel_mean = np.nanmean(vel_all, axis=0)
    edr_mean = np.nanmean(edr_all, axis=0)
    eet_mean = np.nanmean(eet_all, axis=0)
    loss_mean = composed_loss(mse_mean, edr_mean, w_energy)
    mse_std = np.nanstd(mse_all, axis=0)

    # Earliest divergence across starts (ignore the K+1 sentinel).
    finite_divs = [d for d in div_steps if d <= K]
    divergence_step = min(finite_divs) if finite_divs else None

    k = np.arange(1, K + 1, dtype=np.float64)
    mse_slope, mse_r2 = fit_slopes(mse_mean, k)
    edr_slope, edr_r2 = fit_slopes(edr_mean, k)
    loss_slope, loss_r2 = fit_slopes(loss_mean, k)
    log_mse_slope = fit_log_slopes(mse_mean, k)

    # Variant detection. Newer checkpoints carry a `"variant"` key
    # written by the trainers (mlp_train/lstm_train/gnn_train, set from
    # cfg.w_rollout > 0). For older checkpoints missing the key, fall
    # back to the directory-name heuristic (`_stable/` ⇒ stability-
    # trained) so historical results still classify correctly.
    variant = None
    try:
        ckpt_meta = torch.load(ckpt_path, map_location="cpu",
                               weights_only=False)
        variant = ckpt_meta.get("variant")
    except Exception:
        variant = None
    if variant not in ("single_step", "stable"):
        variant = ("stable"
                   if "_stable" in Path(ckpt_path).as_posix()
                   else "single_step")

    def _tolist(a: np.ndarray) -> list:
        return [float(x) if np.isfinite(x) else None for x in a]

    payload = {
        "model_type": model_type,
        "variant": variant,
        "ckpt_path": ckpt_path,
        "n_params": int(n_params),
        "K": K,
        "n_starts": len(starts),
        "rollout_method": rollout_method,
        "per_step": {
            "mse": _tolist(mse_mean),
            "pos_mse": _tolist(pos_mean),
            "vel_mse": _tolist(vel_mean),
            "energy_drift": _tolist(edr_mean),
            "energy_err_vs_true": _tolist(eet_mean),
            "loss": _tolist(loss_mean),
            "mse_std": _tolist(mse_std),
        },
        "gradients": {
            "mse_slope": mse_slope,
            "mse_r2": mse_r2,
            "energy_drift_slope": edr_slope,
            "energy_drift_r2": edr_r2,
            "loss_slope": loss_slope,
            "loss_r2": loss_r2,
            "log_mse_slope": log_mse_slope,
        },
        "divergence_step": divergence_step,
        "start_frames": [[int(s), int(f)] for s, f in starts],
    }
    print(f"  → mse_slope={mse_slope:.4e} (R²={mse_r2:.3f})  "
          f"energy_drift_slope={edr_slope:.4e}  loss_slope={loss_slope:.4e}  "
          f"log_mse_slope={log_mse_slope:.4e}  "
          f"div_step={divergence_step}")
    return payload


# ── Plots ────────────────────────────────────────────────────────────────────
_MODEL_COLORS = {"mlp": "#1f77b4", "lstm": "#2ca02c", "gnn": "#d62728"}


def _label(m: dict) -> str:
    v = " (stable)" if m["variant"] == "stable" else ""
    return f"{m['model_type'].upper()}{v}"


def make_plots(per_model: list[dict], N: int, out_dir: Path) -> None:
    """3-panel figure: MSE (log y), energy drift, composed loss vs k."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    titles = ["Positional+velocity MSE vs step", "Energy drift vs step",
              "Composed loss vs step"]
    ykeys = ["mse", "energy_drift", "loss"]
    logy = [True, False, False]

    for ax, title, key, lg in zip(axes, titles, ykeys, logy):
        for m in per_model:
            y = np.array(m["per_step"][key], dtype=float)
            k = np.arange(1, len(y) + 1)
            mask = np.isfinite(y)
            color = _MODEL_COLORS.get(m["model_type"], "gray")
            ls = "--" if m["variant"] == "stable" else "-"
            ax.plot(k[mask], y[mask], ls, color=color, lw=1.6,
                    label=_label(m))
            if m["divergence_step"] is not None:
                ax.axvline(m["divergence_step"], color=color, ls=":",
                           lw=0.9, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("rollout step k")
        if lg:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if key == "mse":
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Rollout stability, N={N}  (K={per_model[0]['K']}, "
                 f"{per_model[0]['n_starts']} starts)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"stability_N{N}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


def make_overview_plot(all_results: dict, out_dir: Path) -> None:
    """Slopes (mse, energy_drift, loss) vs N, per model type."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not all_results:
        print("[overview] no results to plot.")
        return
    Ns = sorted(all_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes,
                              ["mse_slope", "energy_drift_slope", "loss_slope"],
                              ["MSE slope vs N", "Energy-drift slope vs N",
                               "Loss slope vs N"]):
        for mt in ("mlp", "lstm", "gnn"):
            xs, ys = [], []
            for N in Ns:
                for m in all_results[N]:
                    if m["model_type"] == mt and m["variant"] == "single_step":
                        v = m["gradients"][key]
                        if np.isfinite(v):
                            xs.append(N)
                            ys.append(v)
            if xs:
                ax.plot(xs, ys, "-o", color=_MODEL_COLORS[mt], label=mt.upper())
        ax.set_title(title)
        ax.set_xlabel("N (body count)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Rollout-stability gradients vs N (single-step-trained models)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "stability_overview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Markdown table ───────────────────────────────────────────────────────────
def markdown_table(per_model: list[dict]) -> str:
    head = ("| model | variant | n_params | K | n_starts | mse_slope | mse_r2 | "
            "energy_drift_slope | loss_slope | log_mse_slope | div_step |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for m in per_model:
        g = m["gradients"]
        rows.append(
            f"| {m['model_type'].upper()} | {m['variant']} | {m['n_params']:,} | "
            f"{m['K']} | {m['n_starts']} | {g['mse_slope']:.4e} | {g['mse_r2']:.3f} | "
            f"{g['energy_drift_slope']:.4e} | {g['loss_slope']:.4e} | "
            f"{g['log_mse_slope']:.4e} | {m['divergence_step']} |"
        )
    return head + "\n".join(rows)


# ── Aggregation ──────────────────────────────────────────────────────────────
def aggregate(results_root: Path, out_dir: Path) -> None:
    """Read every results/N*/stability.json → summary JSON + overview plot."""
    paths = sorted(glob.glob(str(results_root / "N*" / "stability.json")))
    if not paths:
        print(f"[aggregate] no results/N*/stability.json found under "
              f"{results_root}")
        return
    all_results: dict[int, list[dict]] = {}
    for p in paths:
        N = int(Path(p).parent.name.replace("N", ""))
        with open(p, "r", encoding="utf-8") as f:
            all_results[N] = json.load(f)

    summary: dict[int, dict] = {}
    for N, models in all_results.items():
        summary[N] = {}
        for m in models:
            key = f"{m['model_type']}_{m['variant']}"
            summary[N][key] = {
                "mse_slope": m["gradients"]["mse_slope"],
                "energy_drift_slope": m["gradients"]["energy_drift_slope"],
                "loss_slope": m["gradients"]["loss_slope"],
                "divergence_step": m["divergence_step"],
                "n_params": m["n_params"],
            }
    out = results_root / "stability_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] {out}")
    print(json.dumps(summary, indent=2))
    make_overview_plot(all_results, out_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_ckpt(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise SystemExit(f"--ckpt must be 'path:type' (got {s!r})")
    path, mtype = s.rsplit(":", 1)
    return path, mtype.strip().lower()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="K-step rollout-stability benchmark with per-step "
                    "MSE / energy-drift / loss curves and slope gradients.")
    p.add_argument("--ckpt", action="append", default=[],
                   help="checkpoint as 'path:type' (type ∈ mlp|lstm|gnn). "
                        "Repeat to compare several models.")
    p.add_argument("--raw-dir", default="raw_data",
                   help="Root holding raw_data/sim_N{N}_XXX.npz trajectories.")
    p.add_argument("--raw-subdir", default=None,
                   help="Force a specific per-model raw subdir, e.g. 'mlp' "
                        "(uses raw_data/N{N}/mlp/). Default: auto-detect "
                        "(root layout, then mlp/lstm/gnn subdirs).")
    p.add_argument("--N", type=int, default=None,
                   help="Body count (required unless --aggregate-only).")
    p.add_argument("--K", type=int, default=128,
                   help="Rollout horizon in steps (default 128).")
    p.add_argument("--rollout-batches", type=int, default=8,
                   help="Number of start frames to average curves over.")
    p.add_argument("--start-frame-stride", type=int, default=500,
                   help="Frame spacing between start frames within a sim.")
    p.add_argument("--n-sims", type=int, default=3,
                   help="Number of raw simulations to sample start frames from.")
    p.add_argument("--json", default=None,
                   help="Output JSON path (list of per-model records).")
    p.add_argument("--out", default="plots", help="Plot output directory.")
    p.add_argument("--eps", type=float, default=DEFAULT_EPS,
                   help="Plummer softening for the energy calculation.")
    p.add_argument("--g", type=float, default=DEFAULT_GRAVITY_G,
                   help="Gravitational constant for the energy calculation.")
    p.add_argument("--w-energy", type=float, default=0.1,
                   help="Weight on energy drift in the composed loss(k).")
    p.add_argument("--blowup-thresh", type=float, default=1e6,
                   help="MSE threshold above which a rollout is marked diverged.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip rollouts; just aggregate results/N*/stability.json "
                        "into a summary + overview plot.")
    p.add_argument("--results-root", default="results",
                   help="Results root for --aggregate-only (default 'results').")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out)

    if args.aggregate_only:
        aggregate(Path(args.results_root), out_dir)
        return

    if not args.ckpt:
        raise SystemExit("no --ckpt given (use 'path:type', repeatable).")
    if args.N is None:
        raise SystemExit("--N is required (or pass --aggregate-only).")

    device = pick_device()
    print(f"[device] {device}")
    raw_root = resolve_raw_root(Path(args.raw_dir), args.N, args.raw_subdir)
    print(f"[raw] using {raw_root} for N={args.N}")

    starts = select_start_frames(raw_root, args.N, args.K, WINDOW_SIZE,
                                 args.rollout_batches, args.start_frame_stride,
                                 args.n_sims)
    print(f"[starts] {len(starts)} start frames for N={args.N}, K={args.K}: "
          f"{starts}")

    per_model: list[dict] = []
    for ckpt in args.ckpt:
        path, mtype = _parse_ckpt(ckpt)
        per_model.append(run_one_model(
            path, mtype, device, raw_root, args.N, args.K, starts,
            args.eps, args.g, args.w_energy, args.blowup_thresh))

    print("\n" + markdown_table(per_model))

    if args.json:
        out_json = Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(per_model, f, indent=2)
        print(f"[json] {out_json}")

    make_plots(per_model, args.N, out_dir)


if __name__ == "__main__":
    main()