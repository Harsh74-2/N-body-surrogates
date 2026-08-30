"""
metrics.py
==========
Real-case validation metrics. Designed to mirror `losses.py` (so the
real-case report uses the same formulas as the synthetic benchmark) and
to add one presets-specific metric: per-preset mean trajectory error
normalised by the system characteristic length L (so the number is
unit-free and comparable across presets of different physical scales).

All inputs are numpy arrays.
"""

from __future__ import annotations

import numpy as np


# ── Trajectory MSE ───────────────────────────────────────────────────────────
def trajectory_mse(surrogate_traj: np.ndarray,
                   reference_traj: np.ndarray) -> float:
    """
    Mean squared error between two trajectories of shape (T, N, 3).

    Averaged over all (t, body, axis) cells. Useful as a coarse
    single-number comparison, but doesn't tell you *where* the
    trajectories diverge, see `trajectory_error_normalised` below.
    """
    if surrogate_traj.shape != reference_traj.shape:
        raise ValueError(
            f"shape mismatch: surrogate {surrogate_traj.shape} vs "
            f"reference {reference_traj.shape}")
    return float(np.mean((surrogate_traj - reference_traj) ** 2))


def trajectory_error_normalised(surrogate_traj: np.ndarray,
                                reference_traj: np.ndarray,
                                char_length: float) -> dict:
    """
    Per-body trajectory error normalised by the system characteristic
    length (so units cancel). Returns a dict:

        per_body_mean : (N,)  : mean |surrogate − reference| / L per body
        per_body_max  : (N,)  : max  |surrogate − reference| / L per body
        per_step_max  : (T,)  : max  |surrogate − reference| / L per step
        frames_before_threshold
                     : int   : first step at which per_step_max > 0.5
                                (i.e. half a system length of error). A
                                project-style "rollout stability" metric
                                expressed in system lengths instead of
                                a fraction of the simulation box.
    """
    if surrogate_traj.shape != reference_traj.shape:
        raise ValueError(
            f"shape mismatch: surrogate {surrogate_traj.shape} vs "
            f"reference {reference_traj.shape}")
    err = np.linalg.norm(surrogate_traj - reference_traj, axis=2)   # (T, N)
    err_norm = err / char_length
    per_body_mean = err_norm.mean(axis=0)
    per_body_max  = err_norm.max(axis=0)
    per_step_max  = err_norm.max(axis=1)
    above = np.where(per_step_max > 0.5)[0]
    frames = int(above[0]) if above.size > 0 else int(per_step_max.size)
    return {
        "per_body_mean":     per_body_mean,
        "per_body_max":      per_body_max,
        "per_step_max":      per_step_max,
        "frames_before_threshold": frames,
        "max_error_over_L":  float(per_step_max.max()),
        "mean_error_over_L": float(err_norm.mean()),
    }


# ── Energy drift (single-step / per-frame) ───────────────────────────────────
def energy_drift_series(positions: np.ndarray,
                        velocities: np.ndarray,
                        mass: np.ndarray,
                        eps: float = 1e-4,
                        g: float = 1.0) -> np.ndarray:
    """
    Compute E(t) for every frame of a trajectory using the upstream
    `simulation_3d.total_energy` formula. Returns shape (T,).
    """
    from simulation_3d import total_energy
    out = np.empty(positions.shape[0], dtype=np.float64)
    for k in range(positions.shape[0]):
        out[k] = total_energy(positions[k], velocities[k], mass, eps, g=g)
    return out


def energy_drift_normalised(E: np.ndarray, eps_floor: float = 1e-8) -> np.ndarray:
    """
    |E(t) − E(0)| / max(|E(0)|, eps_floor) per frame. Returns shape (T,).
    """
    e0 = float(E[0])
    denom = max(abs(e0), eps_floor)
    return np.abs(E - e0) / denom
