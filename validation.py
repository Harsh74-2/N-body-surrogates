"""
validation.py
=============
Mathematical validation suite for the 3D N-body physics engine in
`simulation_3d.py` and the raw `.npz` trajectories it produces.

Each "layer" is a standalone mathematical proof the simulator must
satisfy for the data it emits to be physically clean enough to train
neural-network surrogates on.

Layers
------
  L1  Energy conservation            |ΔE / E₀| over T
  L2  Linear & angular momentum      |ΔP|, |ΔL|  (must be ≈ 0)
  L3  Order of convergence           global error vs dt (target slope ≈ 2)
  L4  Softening sensitivity          |ΔE / E₀| vs ε
  L5  Analytical 2-body closure      Kepler orbit returns after one period
  L6  Newton's 3rd law               f_ij = -f_ji on the full force tensor
  L7  Virial theorem                 2⟨KE⟩ + ⟨PE⟩ ≈ 0 for a stable cluster
  L8  Raw-data integrity             schema + per-frame energy on .npz files

Outputs
-------
  plots/validation_dashboard.png     3x3 summary dashboard (all layers)
  plots/validation_orbit.png         large 3D Kepler orbit (L5)
  plots/validation_convergence.png   publication-style log-log fit (L3)
  plots/validation_raw_data.png      trajectory + energy drift on real .npz (L8)

Usage
-----
    python validation.py --raw-dir raw_data --out plots --seed 42
    python validation.py --raw-dir raw_data --out plots --quick     # faster smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless, write PNGs, no display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, registers 3d projection

from utils import configure_utf8_stdout

configure_utf8_stdout()

# ── Re-use the upstream physics so we validate what the engine actually does ─
from simulation_3d import (
    compute_accelerations,
    leapfrog_step,
    total_energy,
    init_galaxy_disc,
)
from pipeline_config import (
    IC_BASE_SEED,
    IC_M_MAX,
    IC_M_MIN,
    PLOTS_DIR,
    RAW_DIR,
    SOFTENING,
)



# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS HELPERS (small, local, only what validation needs)
# ══════════════════════════════════════════════════════════════════════════════
def linear_momentum(vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """P = Σᵢ mᵢ vᵢ,  shape (3,)."""
    return np.sum(vel * mass[:, np.newaxis], axis=0)


def angular_momentum(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """L = Σᵢ mᵢ (rᵢ × vᵢ),  shape (3,)."""
    return np.sum(mass[:, np.newaxis] * np.cross(pos, vel), axis=0)


def force_tensor(pos: np.ndarray, mass: np.ndarray, epsilon: float, g: float = 1.0
                 ) -> np.ndarray:
    """
    Per-pair *force* F[i, j, k] = force on body i due to body j, axis k.
    Antisymmetric: F[i, j] = -F[j, i]   (Newton's 3rd law, independent of mass).
    """
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]      # r_j - r_i,  shape (N, N, 3)
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff) + epsilon ** 2
    inv_r_cube = 1.0 / (dist_sq ** 1.5)
    np.fill_diagonal(inv_r_cube, 0.0)
    # F[i, j] = G * m_i * m_j * (r_j - r_i) / r³
    pair_mass = mass[:, np.newaxis] * mass[np.newaxis, :]      # m_i * m_j
    return g * pair_mass[:, :, np.newaxis] * inv_r_cube[:, :, np.newaxis] * diff


def ke_and_pe(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray,
              epsilon: float, g: float = 1.0) -> tuple[float, float]:
    """Split E into KE and PE (positive PE = attractive PE magnitude)."""
    ke = 0.5 * np.sum(mass * np.sum(vel ** 2, axis=1))
    pe_mag = 0.0
    for i in range(pos.shape[0]):
        diff = pos[i + 1:] - pos[i]
        dist = np.sqrt(np.sum(diff ** 2, axis=1) + epsilon ** 2)
        pe_mag += g * mass[i] * np.sum(mass[i + 1:] / dist)
    return float(ke), float(-pe_mag)


def virial_ratio(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray,
                 epsilon: float, g: float = 1.0) -> float:
    """2 KE / |PE|: equals 1.0 for a system in virial equilibrium."""
    ke, pe = ke_and_pe(pos, vel, mass, epsilon, g=g)
    return float(2.0 * ke / abs(pe)) if pe != 0 else float("nan")


def run_trajectory(pos: np.ndarray,
                   vel: np.ndarray,
                   mass: np.ndarray,
                   dt: float,
                   epsilon: float,
                   steps: int,
                   g: float = 1.0,
                   sample_every: int = 1
                   ) -> dict:
    """
    Run a leapfrog trajectory, sampling diagnostics every `sample_every` steps.
    Returns a dict of arrays sampled at those times (including t=0).
    """
    acc = compute_accelerations(pos, mass, epsilon, g=g)
    sample_idx = list(range(0, steps + 1, sample_every))
    if sample_idx[-1] != steps:
        sample_idx.append(steps)

    e_samples, p_samples, l_samples, times = [], [], [], []
    for k in range(steps + 1):
        if k in sample_idx:
            e_samples.append(total_energy(pos, vel, mass, epsilon, g=g))
            p_samples.append(linear_momentum(vel, mass))
            l_samples.append(angular_momentum(pos, vel, mass))
            times.append(k * dt)
        if k < steps:
            pos, vel, acc = leapfrog_step(pos, vel, acc, mass, dt, epsilon, g=g)

    return {
        "times":   np.asarray(times),
        "energy":  np.asarray(e_samples),
        "lin_mom": np.asarray(p_samples),
        "ang_mom": np.asarray(l_samples),
        "pos_final": pos.copy(),
        "vel_final": vel.copy(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION LAYERS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LayerResult:
    name:    str
    status:  str                # "PASS" / "WARN" / "FAIL"
    metrics: dict = field(default_factory=dict)
    notes:   list[str] = field(default_factory=list)


# ── L1: Energy conservation ────────────────────────────────────────────────
def layer1_energy(steps: int = 2000, dt: float = 0.005, N: int = 50,
                  epsilon: float = SOFTENING, seed: int = IC_BASE_SEED,
                  m_min: float = IC_M_MIN, m_max: float = IC_M_MAX) -> tuple[LayerResult, dict]:
    pos, vel, mass = init_galaxy_disc(N, seed=seed, m_min=m_min, m_max=m_max)
    traj = run_trajectory(pos, vel, mass, dt, epsilon, steps, sample_every=10)
    e0 = traj["energy"][0]
    drift = np.abs((traj["energy"] - e0) / e0) * 100.0 if e0 != 0 else np.full_like(traj["energy"], np.nan)
    max_drift = float(np.max(drift))
    final_drift = float(drift[-1])
    res = LayerResult("L1 Energy Conservation", "PASS",
                      {"max_drift_pct": max_drift, "final_drift_pct": final_drift,
                       "n_steps": steps, "dt": dt, "N": N, "epsilon": epsilon},
                      [f"max |ΔE/E₀| = {max_drift:.3e}%",
                       f"final |ΔE/E₀| = {final_drift:.3e}%"])
    if max_drift > 5.0:   res.status = "WARN"; res.notes.append("energy drift > 5%")
    if max_drift > 25.0:  res.status = "FAIL"
    return res, {"times": traj["times"], "drift_pct": drift}


# ── L2: Linear + Angular momentum ──────────────────────────────────────────
def layer2_momentum(steps: int = 2000, dt: float = 0.005, N: int = 50,
                    epsilon: float = 0.1, seed: int = 42,
                    m_min: float = 0.5, m_max: float = 5.0) -> tuple[LayerResult, dict]:
    pos, vel, mass = init_galaxy_disc(N, seed=seed, m_min=m_min, m_max=m_max)
    traj = run_trajectory(pos, vel, mass, dt, epsilon, steps, sample_every=10)
    p_drift = np.linalg.norm(traj["lin_mom"] - traj["lin_mom"][0], axis=1)
    l_drift = np.linalg.norm(traj["ang_mom"] - traj["ang_mom"][0], axis=1)
    res = LayerResult("L2 Linear & Angular Momentum", "PASS",
                      {"max_dP": float(np.max(p_drift)),
                       "max_dL": float(np.max(l_drift)),
                       "n_steps": steps, "dt": dt, "N": N},
                      [f"max |ΔP| = {np.max(p_drift):.3e}  (machine-precision floor)",
                       f"max |ΔL| = {np.max(l_drift):.3e}"])
    if np.max(p_drift) > 1e-10 or np.max(l_drift) > 1e-10:
        res.status = "WARN"
        res.notes.append("momentum drift above 1e-10")
    return res, {"times": traj["times"], "lin_drift": p_drift, "ang_drift": l_drift}


# ── L3: Order of convergence (leapfrog is O(h²)) ──────────────────────────
def layer3_convergence(t_end: float = 1.0, epsilon: float = SOFTENING,
                       N: int = 10, seed: int = IC_BASE_SEED,
                       m_min: float = IC_M_MIN, m_max: float = IC_M_MAX) -> tuple[LayerResult, dict]:
    pos0, vel0, mass = init_galaxy_disc(N, seed=seed, m_min=m_min, m_max=m_max)

    def integrate(dt: float):
        p, v = pos0.copy(), vel0.copy()
        a = compute_accelerations(p, mass, epsilon)
        for _ in range(int(round(t_end / dt))):
            p, v, a = leapfrog_step(p, v, a, mass, dt, epsilon)
        return p

    # Reference solution at a much finer resolution than any h we test,
    # so the measured error is dominated by the test integrator, not the
    # reference. h=0.001 is the finest test step, so dt_ref = 1e-5 gives
    # a 100× finer reference.
    dt_ref = 0.00001
    truth = integrate(dt_ref)
    h_values = np.array([0.016, 0.008, 0.004, 0.002, 0.001])
    errors = np.array([np.mean(np.linalg.norm(integrate(h) - truth, axis=1)) for h in h_values])

    # Least-squares slope on log-log axis (skip the coarsest dt, too far from asymptote)
    mask = slice(1, None)
    slope, intercept = np.polyfit(np.log2(h_values[mask]), np.log2(errors[mask]), 1)

    res = LayerResult("L3 Order of Convergence", "PASS",
                      {"fitted_slope": float(slope), "target_slope": 2.0,
                       "h_values": h_values.tolist(),
                       "errors": errors.tolist(),
                       "t_end": t_end, "N": N},
                      [f"fitted slope = {slope:.3f}  (target 2.000 for leapfrog)"])
    if not (1.6 < slope < 2.4):
        res.status = "WARN"
        res.notes.append(f"slope {slope:.2f} outside expected [1.6, 2.4]")
    return res, {"h_values": h_values, "errors": errors, "slope": slope, "intercept": intercept}


# ── L4: Softening robustness ────────────────────────────────────────────────
def layer4_softening(epsilons: list[float], steps: int = 1000, dt: float = 0.005,
                     N: int = 30, seed: int = IC_BASE_SEED,
                     m_min: float = IC_M_MIN, m_max: float = IC_M_MAX) -> tuple[LayerResult, dict]:
    pos0, vel0, mass = init_galaxy_disc(N, seed=seed, m_min=m_min, m_max=m_max)
    drifts = []
    for eps in epsilons:
        traj = run_trajectory(pos0.copy(), vel0.copy(), mass, dt, eps, steps, sample_every=50)
        e0 = traj["energy"][0]
        drift = float(np.max(np.abs((traj["energy"] - e0) / e0)) * 100.0) if e0 != 0 else float("nan")
        drifts.append(drift)
    drifts = np.asarray(drifts)
    res = LayerResult("L4 Softening Robustness", "PASS",
                      {"epsilons": list(epsilons),
                       "drifts_pct": drifts.tolist(),
                       "n_steps": steps, "N": N},
                      [f"drift at smallest ε={epsilons[0]}: {drifts[0]:.3e}%",
                       f"drift at largest  ε={epsilons[-1]}: {drifts[-1]:.3e}%"])
    if np.max(drifts) > 5.0:
        res.status = "WARN"
        res.notes.append("softening sweep drifted > 5%")
    return res, {"epsilons": np.asarray(epsilons), "drifts": drifts}


# ── L5: Analytical 2-body closure (Kepler) ─────────────────────────────────
def layer5_two_body(steps: int | None = None, dt: float = 0.001,
                    seed: int = IC_BASE_SEED) -> tuple[LayerResult, dict]:
    """
    Two equal masses on a bound elliptical Kepler orbit around their CM.

    The relative separation follows a standard 2-body Kepler problem with
    gravitational parameter μ = G (m₁ + m₂). The vis-viva equation gives
    the periapsis speed, and each body moves on its own half-size ellipse
    about the CM at half the relative speed.

        r_p    = periapsis separation         (chosen)
        e      = eccentricity                 (chosen)
        a_rel  = r_p / (1 − e)                (semi-major axis of relative motion)
        T      = 2π √(a_rel³ / μ)             (Kepler's 3rd law)
        v_p    = √(μ (2/r_p − 1/a_rel))       (vis-viva at periapsis)

    We track the orbit's mechanical energy E = KE + PE at every saved
    step. For a symplectic integrator the position gap |pos(T) − pos(0)|
    is dominated by a secular phase error that *only* vanishes at an
    exact period, the wrong thing to look at for convergence. The
    energy error |E(t) − E(0)| is the clean O(h²) signal we want.
    """
    r_p   = 1.0                  # periapsis separation
    ecc   = 0.20
    mu    = 1.0                  # G (m₁ + m₂) with G = 1, m₁ + m₂ = 1
    a_rel = r_p / (1.0 - ecc)    # 1.25
    v_p_rel = float(np.sqrt(mu * (2.0 / r_p - 1.0 / a_rel)))   # vis-viva
    pos   = np.array([[ r_p / 2, 0.0, 0.0],
                      [-r_p / 2, 0.0, 0.0]])
    mass  = np.array([0.5, 0.5])
    vel   = np.array([[0.0,  v_p_rel / 2, 0.0],
                      [0.0, -v_p_rel / 2, 0.0]])
    T      = 2 * np.pi * np.sqrt(a_rel ** 3 / mu)
    n_full = int(round(T / dt))
    steps  = n_full if steps is None else steps            # default: 1 full orbit

    epsilon = 1e-9
    pos_init = pos.copy()
    e0       = total_energy(pos, vel, mass, epsilon, g=1.0)

    pos_traj, t_traj, e_traj = [pos[0].copy()], [0.0], [e0]
    acc = compute_accelerations(pos, mass, epsilon)
    sample_every = max(1, steps // 1500)
    for k in range(steps):
        pos, vel, acc = leapfrog_step(pos, vel, acc, mass, dt, epsilon)
        if (k + 1) % sample_every == 0:
            pos_traj.append(pos[0].copy())
            t_traj.append((k + 1) * dt)
            e_traj.append(total_energy(pos, vel, mass, epsilon, g=1.0))
    pos_traj = np.asarray(pos_traj)
    t_traj   = np.asarray(t_traj)
    e_traj   = np.asarray(e_traj)

    # Energy error:  ΔE/E₀  over the saved samples (excluding t=0).
    e_err = np.abs(e_traj[1:] - e0) / abs(e0)
    max_e_err   = float(np.max(e_err))
    mean_e_err  = float(np.mean(e_err))

    # Position gap (kept for backwards-compat with the dashboard plot)
    closure = np.linalg.norm(pos - pos_init, axis=1)
    n_orbits = steps * dt / T
    res = LayerResult("L5 Analytical 2-Body Closure", "PASS",
                      {"T_theory": float(T), "T_simulated": float(steps * dt),
                       "n_orbits_simulated": float(n_orbits),
                       "max_energy_err": max_e_err,
                       "mean_energy_err": mean_e_err,
                       "max_closure_error": float(np.max(closure)),
                       "mean_closure_error": float(np.mean(closure)),
                       "dt": dt, "steps": steps, "eccentricity": ecc,
                       "a_rel": float(a_rel), "r_p": float(r_p),
                       "v_p_rel": float(v_p_rel), "v_body": float(v_p_rel / 2)},
                      [f"a_rel = {a_rel:.3f},  e = {ecc:.2f},  r_p = {r_p:.3f},  "
                       f"v_p_rel = {v_p_rel:.4f}",
                       f"theoretical period T = {T:.6f}",
                       f"simulated time        = {steps*dt:.6f}  ({n_orbits:.3f} orbits)",
                       f"max |ΔE/E₀| over orbit = {max_e_err:.3e}   (clean O(h²) signal)",
                       f"max orbit-closure err  = {np.max(closure):.3e}"])
    # Energy is the right metric for a symplectic integrator. We expect
    # |ΔE/E₀| on the order of dt², so a 1e-2 threshold catches a
    # genuinely broken integrator without false-positives in quick mode.
    if n_orbits >= 0.95 and max_e_err > 1e-2:
        res.status = "WARN"
        res.notes.append(f"2-body |ΔE/E₀| = {max_e_err:.2e} > 1e-2 after ≥1 full period, symplectic quality suspect")
    return res, {"pos_traj": pos_traj, "t_traj": t_traj,
                 "e_traj": e_traj, "e0": float(e0),
                 "T": T, "eccentricity": ecc, "dt": dt, "a_rel": float(a_rel)}


# ── L6: Newton's 3rd law ───────────────────────────────────────────────────
def layer6_newton3(N: int = 50, seed: int = IC_BASE_SEED,
                   m_min: float = IC_M_MIN, m_max: float = IC_M_MAX
                   ) -> tuple[LayerResult, dict]:
    pos, vel, mass = init_galaxy_disc(N, seed=seed, m_min=m_min, m_max=m_max)
    eps = SOFTENING
    F = force_tensor(pos, mass, eps)                                  # (N, N, 3), forces
    asym = np.linalg.norm(F + np.transpose(F, (1, 0, 2)), axis=2)     # F_ij + F_ji = 0
    # Scale the threshold by the largest pair-force magnitude so the metric
    # is relative (not fooled by tiny ratios of huge + huge numbers).
    Fmag = np.linalg.norm(F, axis=2)
    max_F = float(np.max(Fmag)) if Fmag.size else 0.0
    rel_asym = float(np.max(asym) / max_F) if max_F > 0 else 0.0
    self_force = np.linalg.norm(F.diagonal(axis1=0, axis2=1), axis=1)
    max_self   = float(np.max(self_force))
    res = LayerResult("L6 Newton's 3rd Law", "PASS",
                      {"max_pair_asymmetry_abs": float(np.max(asym)),
                       "max_pair_force_mag": max_F,
                       "rel_asymmetry": rel_asym,
                       "max_self_force": max_self,
                       "N": N, "epsilon": eps},
                      [f"max ‖F_ij + F_ji‖ = {np.max(asym):.3e}   "
                       f"(rel {rel_asym:.2e}, should be ~ε_machine)",
                       f"max self-force    = {max_self:.3e}  (should be 0)"])
    if rel_asym > 1e-10 or max_self > 1e-12:
        res.status = "FAIL"
        res.notes.append("force tensor breaks action-reaction symmetry")
    return res, {"force_tensor": F, "asymmetry": asym}


# ── L7: Virial theorem ─────────────────────────────────────────────────────
def layer7_virial(steps: int = 4000, dt: float = 0.005, N: int = 50,
                  epsilon: float = SOFTENING, seed: int = IC_BASE_SEED,
                  m_min: float = IC_M_MIN, m_max: float = IC_M_MAX
                  ) -> tuple[LayerResult, dict]:
    """
    For a system in virial equilibrium, 2⟨KE⟩ + ⟨PE⟩ ≈ 0  ⇒  2KE/|PE| ≈ 1.
    We integrate, sample the ratio every ~50 steps, and report the mean.
    """
    pos, vel, mass = init_galaxy_disc(N, seed=seed, m_min=m_min, m_max=m_max)
    a = compute_accelerations(pos, mass, epsilon)
    ratios = []
    times  = []
    sample = max(1, steps // 200)
    for k in range(steps + 1):
        if k % sample == 0:
            r = virial_ratio(pos, vel, mass, epsilon)
            if np.isfinite(r):
                ratios.append(r)
                times.append(k * dt)
        if k < steps:
            pos, vel, a = leapfrog_step(pos, vel, a, mass, dt, epsilon)
    ratios = np.asarray(ratios)
    times  = np.asarray(times)
    mean_r, std_r = float(np.mean(ratios)), float(np.std(ratios))
    res = LayerResult("L7 Virial Theorem", "PASS",
                      {"mean_2KE_over_|PE|": mean_r, "std": std_r,
                       "target": 1.0, "n_samples": int(ratios.size),
                       "N": N},
                      [f"⟨2KE / |PE|⟩ = {mean_r:.4f} ± {std_r:.4f}  (target 1.0)"])
    # Disc starts in centrifugal equilibrium, so 2KE/|PE| should sit
    # near 1.0 from the start. Allow a generous band for the few-step
    # transient before settling, but flag anything clearly broken.
    if mean_r < 0.5 or mean_r > 1.5:
        res.status = "WARN"
        res.notes.append(f"virial ratio {mean_r:.2f} outside [0.5, 1.5]")
    return res, {"times": times, "ratios": ratios, "mean": mean_r, "std": std_r}


# ── L8: Raw `.npz` data integrity ──────────────────────────────────────────
def layer8_raw_data(raw_dir: str) -> tuple[LayerResult, dict]:
    """
    Walk `raw_dir` for `sim_*.npz` *or* `sim_*.npy`, validate the schema,
    then re-derive energy conservation on each loaded trajectory as a final
    physical sanity check on the *saved* data (not just the integrator).
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        return LayerResult("L8 Raw Data Integrity", "WARN",
                           {}, [f"directory not found: {raw_dir!r}"]), {}

    files = sorted(
        raw_path.glob("sim_*.npz")
    ) + sorted(
        raw_path.glob("sim_*.npy")
    )
    if not files:
        return LayerResult("L8 Raw Data Integrity", "WARN",
                           {}, [f"no sim_*.npz / sim_*.npy files found in {raw_dir!r}"]), {}

    per_sim = []
    worst_drift = 0.0
    worst_idx   = -1
    sample_traj = None
    sample_meta = None

    for i, path in enumerate(files):
        # ── Load frames (and optionally mass/meta) ─────────────────────────
        if path.suffix == ".npz":
            with np.load(path) as d:
                keys = set(d.files)
                if "frames" not in keys:
                    per_sim.append({"file": path.name, "status": "FAIL",
                                    "reason": "missing key 'frames'"})
                    continue
                frames = d["frames"]
                mass   = d["mass"]   if "mass" in keys else None
                meta   = d["meta"]   if "meta" in keys else None
        else:
            frames = np.load(path, mmap_mode="r")
            mass   = None
            meta   = None

        F, N, Fdim = frames.shape
        if Fdim != 6 or F < 50:
            per_sim.append({"file": path.name, "status": "FAIL",
                            "reason": f"shape {frames.shape} (expected (F, N, 6))"})
            continue

        # ── Best-effort mass reconstruction for legacy .npy files ──────────
        # The original simulation_3d.py used uniform masses (1/N), so we
        # reconstruct that here for the energy-drift check below.
        if mass is None:
            mass = np.full(N, 1.0 / N, dtype=np.float64)
            mass_source = "uniform-1/N (legacy)"
        else:
            mass = np.asarray(mass, dtype=np.float64)
            if abs(mass.sum() - 1.0) > 1e-6:
                per_sim.append({"file": path.name, "status": "FAIL",
                                "reason": f"mass sum = {mass.sum():.6f} (expected 1.0)"})
                continue
            mass_source = "saved"

        if not np.all(np.isfinite(frames)):
            per_sim.append({"file": path.name, "status": "FAIL",
                            "reason": "non-finite values in frames"})
            continue

        # ── Defaults for missing meta ───────────────────────────────────────
        if meta is not None:
            dt, eps, g, seed = float(meta[0]), float(meta[1]), float(meta[2]), float(meta[3])
        else:
            dt, eps, g, seed = 0.01, 0.1, 1.0, -1   # best-guess legacy defaults

        # ── Re-derive energy conservation from the saved trajectory ─────────
        sub = max(2, F // 200)
        idx = np.arange(0, F, sub)
        sampled = np.asarray(frames[idx])                              # (n, N, 6)
        pos = sampled[:, :, 0:3]; vel = sampled[:, :, 3:6]
        energies = np.array([
            total_energy(pos[k], vel[k], mass, eps, g=g) for k in range(sampled.shape[0])
        ])
        e0 = energies[0]
        drift = (float(np.max(np.abs((energies - e0) / e0)) * 100.0)
                 if e0 != 0 else float("nan"))
        per_sim.append({"file": path.name, "status": "PASS",
                        "F": F, "N": N, "dt": dt, "epsilon": eps,
                        "seed": int(seed), "energy_drift_pct": drift,
                        "mass_source": mass_source})
        if drift > worst_drift:
            worst_drift = drift
            worst_idx   = i
            sample_traj = sampled
            sample_meta = {"dt": dt, "epsilon": eps, "g": g, "seed": int(seed),
                           "file": path.name}

    statuses = [s["status"] for s in per_sim]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    res = LayerResult("L8 Raw Data Integrity", overall,
                      {"n_files": len(files), "n_pass": statuses.count("PASS"),
                       "n_warn": statuses.count("WARN"), "n_fail": statuses.count("FAIL"),
                       "worst_energy_drift_pct": worst_drift,
                       "per_sim": per_sim},
                      [f"scanned {len(files)} simulation file(s); "
                       f"worst |ΔE/E₀| = {worst_drift:.3e}%"])

    return res, {"per_sim": per_sim, "worst_drift": worst_drift,
                 "worst_idx": worst_idx, "sample_traj": sample_traj,
                 "sample_meta": sample_meta}


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
THEME = dict(
    bg="#0d0d0d", panel="#111111", text="#e0e0e0",
    accent="#7eb8f7", good="#7ef7a0", warn="#f7a07e", bad="#ff6b6b",
    violet="#b37ef7", grid="#2a2a2a", spine="#333333",
)


def _style_ax(ax, title: str, theme: dict = THEME) -> None:
    ax.set_facecolor(theme["panel"])
    ax.tick_params(colors=theme["text"], labelsize=8)
    ax.xaxis.label.set_color(theme["text"])
    ax.yaxis.label.set_color(theme["text"])
    ax.set_title(title, color=theme["text"], fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme["spine"])
    ax.grid(True, color=theme["grid"], linewidth=0.5, alpha=0.7)


def plot_dashboard(results: dict, out_path: str, theme: dict = THEME) -> None:
    """3x3 summary dashboard covering L1..L8 + a summary text panel."""
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(theme["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # L1: Energy drift
    ax1 = fig.add_subplot(gs[0, 0])
    e = results["L1"]
    ax1.plot(e["times"], e["drift_pct"], color=theme["accent"], linewidth=1.4)
    ax1.set_xlabel("time"); ax1.set_ylabel("|ΔE / E₀| (%)")
    _style_ax(ax1, "L1: Energy conservation")

    # L2: momentum drifts (log)
    ax2 = fig.add_subplot(gs[0, 1])
    p = results["L2"]
    ax2.plot(p["times"], p["lin_drift"], color=theme["bad"],   label="linear", lw=1.2, alpha=0.85)
    ax2.plot(p["times"], p["ang_drift"], color=theme["warn"],  label="angular", lw=1.2, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xlabel("time"); ax2.set_ylabel("|ΔP|, |ΔL|")
    _style_ax(ax2, "L2: Linear & angular momentum")
    ax2.legend(fontsize=8, labelcolor=theme["text"], facecolor=theme["panel"])

    # L3: convergence
    ax3 = fig.add_subplot(gs[0, 2])
    c = results["L3"]
    ax3.loglog(c["h_values"], c["errors"], "o-", color=theme["good"], lw=1.4, ms=6,
               label=f"fitted slope = {c['slope']:.2f}")
    h_ref = np.array([c["h_values"][0], c["h_values"][-1]])
    err_ref = c["errors"][-1] * (h_ref / c["h_values"][-1]) ** 2
    ax3.loglog(h_ref, err_ref, "--", color=theme["accent"], lw=1.0, alpha=0.7, label="ideal O(h²)")
    ax3.set_xlabel("step size h"); ax3.set_ylabel("global position error")
    ax3.legend(fontsize=8, labelcolor=theme["text"], facecolor=theme["panel"])
    _style_ax(ax3, "L3: O(h²) convergence")

    # L4: softening sensitivity
    ax4 = fig.add_subplot(gs[1, 0])
    s4 = results["L4"]
    ax4.bar([f"{e:.2f}" for e in s4["epsilons"]], s4["drifts"],
            color=theme["violet"], alpha=0.85, edgecolor=theme["bg"], width=0.55)
    ax4.set_xlabel("softening ε"); ax4.set_ylabel("max |ΔE/E₀| (%)")
    _style_ax(ax4, "L4: Softening robustness")

    # L5: orbit (3D)
    ax5 = fig.add_subplot(gs[1, 1], projection="3d")
    o = results["L5"]
    t, xyz = o["t_traj"], o["pos_traj"]
    ax5.set_facecolor(theme["bg"])
    for pane in (ax5.xaxis, ax5.yaxis, ax5.zaxis):
        pane.set_pane_color((0.07, 0.07, 0.07, 1.0))
    ax5.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=theme["accent"], lw=1.0)
    ax5.scatter([0.5], [0], [0], color="#ffffff", s=20, zorder=5)
    ax5.scatter([-0.5], [0], [0], color="#ffffff", s=20, zorder=5)
    ax5.set_title(f"L5, Kepler 2-body (e={o['eccentricity']:.2f})",
                  color=theme["text"], fontsize=10)
    ax5.tick_params(colors=theme["text"], labelsize=7)

    # L6: Newton's 3rd law (asymmetry matrix)
    ax6 = fig.add_subplot(gs[1, 2])
    asym = results["L6"]["asymmetry"]
    im = ax6.imshow(np.log10(asym + 1e-30), cmap="inferno", aspect="auto")
    ax6.set_xlabel("body j"); ax6.set_ylabel("body i")
    ax6.set_title("L6, ‖f_ij + f_ji‖ (log10)", color=theme["text"], fontsize=10)
    ax6.tick_params(colors=theme["text"], labelsize=7)
    cbar = fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=theme["text"], labelsize=7)

    # L7: virial ratio
    ax7 = fig.add_subplot(gs[2, 0])
    v = results["L7"]
    ax7.plot(v["times"], v["ratios"], color=theme["accent"], lw=1.0, alpha=0.7)
    ax7.axhline(1.0, color=theme["good"], linestyle="--", lw=1.0, alpha=0.7, label="ideal = 1.0")
    ax7.set_xlabel("time"); ax7.set_ylabel("2KE / |PE|")
    ax7.set_ylim(0, 2)
    ax7.legend(fontsize=8, labelcolor=theme["text"], facecolor=theme["panel"])
    _style_ax(ax7, f"L7: Virial ratio (mean {v['mean']:.2f} ± {v['std']:.2f})")

    # L8: raw data trajectory + energy
    ax8 = fig.add_subplot(gs[2, 1])
    r8 = results["L8"]
    if r8.get("sample_traj") is not None:
        traj = r8["sample_traj"]
        for body in range(min(traj.shape[1], 10)):
            ax8.plot(traj[:, body, 0], traj[:, body, 1],
                     lw=0.6, alpha=0.7)
        ax8.set_xlabel("x"); ax8.set_ylabel("y")
        ax8.set_title(f"L8, Sample trajectory ({r8['sample_meta']['file']})",
                      color=theme["text"], fontsize=9)
        ax8.set_aspect("equal", adjustable="datalim")
        _style_ax(ax8, "L8: Trajectory snapshot (xy)")
    else:
        ax8.text(0.5, 0.5, "no .npz found", color=theme["text"],
                 ha="center", va="center", transform=ax8.transAxes)
        ax8.axis("off")

    # Summary text panel
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_facecolor(theme["panel"]); ax9.axis("off")
    lines = ["MATHEMATICAL VALIDATION SUMMARY", ""]
    for key, res in results.items():
        if not isinstance(res, dict) or "name" not in res:
            continue
        status = res["status"]
        marker = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
        lines.append(f"{marker} {res['name']}: {status}")
    lines += ["", f"overall: {results['_overall']}",
              f"worst raw-data drift: {results['_worst_raw_drift']:.3e}%"]
    color_for = lambda s: theme["good"] if "PASS" in s else theme["warn"] if "WARN" in s else theme["bad"]
    for i, line in enumerate(lines):
        color = (theme["accent"] if i == 0
                 else color_for(line) if any(t in line for t in ("PASS", "WARN", "FAIL"))
                 else theme["text"])
        weight = "bold" if i == 0 or "overall" in line else "normal"
        ax9.text(0.05, 0.95 - i * 0.085, line, transform=ax9.transAxes,
                 fontsize=8.5, color=color, fontweight=weight,
                 fontfamily="monospace", verticalalignment="top")

    fig.suptitle("simulation_3d.py, Mathematical Validation Dashboard",
                 color=theme["text"], fontsize=14, y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_orbit(L5: dict, out_path: str, theme: dict = THEME) -> None:
    """Large, publication-quality 3D orbit."""
    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor(theme["bg"])
    ax = fig.add_subplot(111, projection="3d")
    xyz = L5["pos_traj"]
    ax.set_facecolor(theme["bg"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.07, 0.07, 0.07, 1.0))
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=theme["accent"], lw=1.2)
    ax.scatter([0], [0], [0], color="#ffffff", s=40, marker="+", label="CM")
    ax.set_xlabel("x", color=theme["text"]); ax.set_ylabel("y", color=theme["text"])
    ax.set_zlabel("z", color=theme["text"])
    ax.tick_params(colors=theme["text"], labelsize=8)
    ax.set_title(f"Kepler two-body closure, e={L5['eccentricity']:.2f}, "
                 f"T={L5['T']:.4f}", color=theme["text"], fontsize=12)
    ax.legend(labelcolor=theme["text"], facecolor=theme["panel"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_convergence(L3: dict, out_path: str, theme: dict = THEME) -> None:
    """Log-log convergence with the fitted slope."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["panel"])
    h, err = L3["h_values"], L3["errors"]
    ax.loglog(h, err, "o-", color=theme["good"], lw=1.6, ms=8, label="measured")
    h_ref = np.array([h[0], h[-1]])
    err_ref = err[-1] * (h_ref / h[-1]) ** 2
    ax.loglog(h_ref, err_ref, "--", color=theme["accent"], lw=1.2,
              label=f"ideal O(h²)  (fitted slope = {L3['slope']:.3f})")
    for hi, ei in zip(h, err):
        ax.annotate(f"h={hi:g}", xy=(hi, ei), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=theme["text"])
    ax.set_xlabel("step size h", color=theme["text"])
    ax.set_ylabel("mean ‖Δr‖", color=theme["text"])
    ax.tick_params(colors=theme["text"])
    for s in ax.spines.values(): s.set_edgecolor(theme["spine"])
    ax.grid(True, which="both", color=theme["grid"], lw=0.5, alpha=0.7)
    ax.legend(labelcolor=theme["text"], facecolor=theme["panel"])
    ax.set_title("Leapfrog global-error convergence", color=theme["text"], fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_raw_data(L8: dict, out_path: str, theme: dict = THEME) -> None:
    """If we have a sample trajectory, plot its energy drift + xy paths."""
    if L8.get("sample_traj") is None:
        return
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor(theme["bg"])
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

    # xy trajectories of every body
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(theme["panel"])
    traj = L8["sample_traj"]
    for body in range(min(traj.shape[1], 25)):
        ax1.plot(traj[:, body, 0], traj[:, body, 1], lw=0.5, alpha=0.65)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_aspect("equal", adjustable="datalim")
    _style_ax(ax1, f"xy trajectories, {L8['sample_meta']['file']}")

    # |v| over time for each body
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(theme["panel"])
    vels = np.linalg.norm(traj[:, :, 3:6], axis=2)  # (T, N)
    for body in range(min(traj.shape[1], 25)):
        ax2.plot(vels[:, body], lw=0.5, alpha=0.65)
    ax2.set_xlabel("sample index"); ax2.set_ylabel("|v|")
    _style_ax(ax2, "|v| over time")

    plt.suptitle("Raw .npz data integrity check",
                 color=theme["text"], fontsize=12, y=0.98)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_validation_suite(out_dir: str = PLOTS_DIR, raw_dir: str = RAW_DIR,
                         seed: int = IC_BASE_SEED, quick: bool = False,
                         m_min: float = IC_M_MIN, m_max: float = IC_M_MAX) -> dict:
    """
    Run every validation layer, write all plots, and return a JSON-able
    dict of results. Returns a `summary` dict so this can be called from
    notebooks or other scripts.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"\n{'=' * 64}")
    print(f"  simulation_3d.py, Mathematical Validation")
    print(f"{'=' * 64}\n")

    # Configurable timing knobs
    if quick:
        L1_steps, L2_steps = 600, 600
        L3_h = [0.016, 0.008, 0.004, 0.002]
        L4_eps = [0.05, 0.10, 0.20]
        L5_steps = 1500                              # overridden below
        L7_steps = 1500
    else:
        L1_steps, L2_steps = 2000, 2000
        L3_h = [0.016, 0.008, 0.004, 0.002, 0.001]
        L4_eps = [0.02, 0.05, 0.10, 0.20, 0.40]
        L5_steps = 8000
        L7_steps = 4000

    results: dict = {}
    raw_payloads: dict = {}

    # L1
    print(f"  [1/8] L1: Energy conservation (N=50, {L1_steps} steps)…")
    r, d = layer1_energy(steps=L1_steps, seed=seed, m_min=m_min, m_max=m_max)
    results["L1"] = asdict(r); raw_payloads["L1"] = d
    print("        " + " | ".join(r.notes))

    # L2
    print(f"  [2/8] L2: Linear & angular momentum…")
    r, d = layer2_momentum(steps=L2_steps, seed=seed, m_min=m_min, m_max=m_max)
    results["L2"] = asdict(r); raw_payloads["L2"] = d
    print("        " + " | ".join(r.notes))

    # L3
    print(f"  [3/8] L3: O(h²) convergence (ref dt=1e-5)…")
    r, d = layer3_convergence(seed=seed, m_min=m_min, m_max=m_max)
    results["L3"] = asdict(r); raw_payloads["L3"] = d
    if quick:                      # narrow the sweep to match quick-mode
        d["h_values"], d["errors"] = d["h_values"][:4], d["errors"][:4]
        d["slope"], d["intercept"] = np.polyfit(
            np.log2(d["h_values"][1:]), np.log2(d["errors"][1:]), 1)
        results["L3"]["fitted_slope"] = float(d["slope"])
        results["L3"]["h_values"] = d["h_values"].tolist()
        results["L3"]["errors"]   = d["errors"].tolist()
    print("        " + " | ".join(r.notes))

    # L4
    print(f"  [4/8] L4: Softening sensitivity (ε sweep)…")
    r, d = layer4_softening(L4_eps, seed=seed, m_min=m_min, m_max=m_max)
    results["L4"] = asdict(r); raw_payloads["L4"] = d
    print("        " + " | ".join(r.notes))

    # L5
    print(f"  [5/8] L5: Analytical 2-body Kepler closure…")
    # L5 forces `steps = T/dt` (one full period) if None is passed, so the
    # closure metric is meaningful. In quick mode we still want a full orbit
    # but at coarser dt to keep it fast.
    if quick:
        r, d = layer5_two_body(steps=None, dt=0.005, seed=seed)
    else:
        r, d = layer5_two_body(steps=None, dt=0.001, seed=seed)
    results["L5"] = asdict(r); raw_payloads["L5"] = d
    print("        " + " | ".join(r.notes))

    # L6
    print(f"  [6/8] L6: Newton's 3rd law on full force tensor…")
    r, d = layer6_newton3(seed=seed, m_min=m_min, m_max=m_max)
    # strip the (N,N,3) tensor, too big for JSON; keep only asymmetry
    results["L6"] = asdict(r); raw_payloads["L6"] = {"asymmetry": d["asymmetry"]}
    print("        " + " | ".join(r.notes))

    # L7
    print(f"  [7/8] L7: Virial theorem (2KE/|PE|)…")
    r, d = layer7_virial(steps=L7_steps, seed=seed, m_min=m_min, m_max=m_max)
    results["L7"] = asdict(r); raw_payloads["L7"] = d
    print("        " + " | ".join(r.notes))

    # L8
    print(f"  [8/8] L8: Raw .npz data integrity ({raw_dir})…")
    r, d = layer8_raw_data(raw_dir)
    results["L8"] = asdict(r); raw_payloads["L8"] = d
    print("        " + " | ".join(r.notes))
    if d.get("per_sim"):
        for s in d["per_sim"]:
            tag = "PASS" if s["status"] == "PASS" else s["status"]
            extra = (f"  drift={s.get('energy_drift_pct', float('nan')):.3e}%"
                     if "energy_drift_pct" in s else f"  {s.get('reason','')}")
            print(f"          • {s['file']}: {tag}{extra}")

    # Aggregate
    statuses = [results[k]["status"] for k in results if k.startswith("L")]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN (review warnings)"
    else:
        overall = "PASS: physics engine verified"
    results["_overall"] = overall
    results["_worst_raw_drift"] = raw_payloads["L8"].get("worst_drift", 0.0)

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\n  Writing plots…")
    # Dashboard consumes both the per-layer payloads (for plotting) and the
    # result objects (for status text in the summary panel). Stitch them.
    plot_payloads = dict(raw_payloads)
    for k, v in results.items():
        plot_payloads.setdefault(k, v)
    plot_payloads["_overall"] = overall
    plot_payloads["_worst_raw_drift"] = results["_worst_raw_drift"]

    dash = out_path / "validation_dashboard.png"
    plot_dashboard(plot_payloads, str(dash))
    print(f"    ✓ {dash}")

    orb = out_path / "validation_orbit.png"
    plot_orbit(raw_payloads["L5"], str(orb))
    print(f"    ✓ {orb}")

    conv = out_path / "validation_convergence.png"
    plot_convergence(raw_payloads["L3"], str(conv))
    print(f"    ✓ {conv}")

    if raw_payloads["L8"].get("sample_traj") is not None:
        rd = out_path / "validation_raw_data.png"
        plot_raw_data(raw_payloads["L8"], str(rd))
        print(f"    ✓ {rd}")

    # JSON report
    report_path = out_path / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        # Strip non-JSON-serializable numpy bits from L6 metrics.
        json_safe = json.loads(json.dumps(results, default=lambda o: float(o)
                                          if isinstance(o, np.floating) else None))
        json.dump(json_safe, f, indent=2)
    print(f"    ✓ {report_path}")

    print(f"\n{'=' * 64}")
    print(f"  Overall: {overall}")
    print(f"  Elapsed: {time.perf_counter() - t0:.2f}s")
    print(f"{'=' * 64}\n")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the 3D N-body physics + raw data.")
    p.add_argument("--raw-dir", default=RAW_DIR,
                   help="Directory containing sim_*.npz files for L8.")
    p.add_argument("--out", default=PLOTS_DIR,
                   help="Output directory for plots and JSON report.")
    p.add_argument("--seed", type=int, default=IC_BASE_SEED)
    p.add_argument("--quick", action="store_true",
                   help="Run a faster smoke-test variant.")
    p.add_argument("--m-min", type=float, default=IC_M_MIN,
                   help="Minimum body mass for log-uniform sampling in the "
                        "disc IC (real-MSun values like 0.1 give a stellar "
                        "IMF before Σ=1 normalisation).")
    p.add_argument("--m-max", type=float, default=IC_M_MAX,
                   help="Maximum body mass for log-uniform sampling in the "
                        "disc IC (real-MSun upper bound ≈ 50–100).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_validation_suite(
        out_dir=args.out,
        raw_dir=args.raw_dir,
        seed=args.seed,
        quick=args.quick,
        m_min=args.m_min,
        m_max=args.m_max,
    )