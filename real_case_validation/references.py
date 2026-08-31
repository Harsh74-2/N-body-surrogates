"""
references.py
=============
Ground-truth integrators used to evaluate the trained surrogates on
real Solar-System scenarios.

Two references are shipped:

1. `reference_leapfrog`, wraps `simulation_3d.leapfrog_step` at a
   sub-stepped dt (`dt_ref = sim_dt / ref_substeps`). The project's
   `validation.py` shows the leapfrog at this resolution stays inside
   ~1e-3 % energy drift for a 2000-step run; that's a comfortably
   tight reference for the real-case comparison.

2. `reference_kepler`, closed-form 2-body Keplerian orbit, used for
   the Sun-Earth-only preset so we have a true analytical reference
   (not "high-precision numerical") for the simplest case.

Both return numpy arrays with the same layout, so the rest of the
runner is reference-agnostic.
"""

from __future__ import annotations

import math

import numpy as np

# Reuse the upstream physics so the reference is exactly what
# `validation.py` validated. We pull leapfrog_step + compute_accelerations
# + total_energy directly from the project module.
from simulation_3d import (
    compute_accelerations,
    leapfrog_step,
    total_energy,
)
from pipeline_config import DEFAULT_GRAVITY_G


def reference_leapfrog(pos0: np.ndarray,
                       vel0: np.ndarray,
                       mass: np.ndarray,
                       dt_N: float,
                       n_steps: int,
                       epsilon: float = 1e-4,
                       g: float = DEFAULT_GRAVITY_G,
                       ref_substeps: int = 100,
                       sample_every: int | None = None,
                       ) -> dict:
    """
    Run a sub-stepped leapfrog reference from (pos0, vel0, mass).

    Parameters
    ----------
    pos0, vel0    : (N, 3): initial state in N-body units
    mass          : (N,)  : per-body mass (Σ = 1)
    dt_N          : float , *coarse* time step per sample (the same dt
                             the surrogate consumes)
    n_steps       : int   : total number of coarse samples to record
    epsilon       : float , Plummer softening in N-body units.
                             Default 1e-4 is tighter than the training
                             default 0.1 to keep the reference clean
                             in cases where the rescaling maps a
                             physical ε to something large.
    ref_substeps  : int   : how many fine leapfrog steps per coarse dt
    sample_every  : int   : record every k-th coarse sample (defaults
                             to 1 = record every sample).

    Returns
    -------
    dict with keys:
        pos : (n_recorded, N, 3)
        vel : (n_recorded, N, 3)
        energy : (n_recorded,) , total mechanical energy per frame
        t : (n_recorded,)       , dimensionless time per frame
    """
    if sample_every is None:
        sample_every = 1

    dt_fine = dt_N / ref_substeps
    pos, vel = pos0.copy(), vel0.copy()
    acc = compute_accelerations(pos, mass, epsilon, g=g)

    pos_list, vel_list, e_list, t_list = [], [], [], []
    # t=0 sample
    pos_list.append(pos.copy())
    vel_list.append(vel.copy())
    e_list.append(total_energy(pos, vel, mass, epsilon, g=g))
    t_list.append(0.0)

    step_idx = 0
    for k in range(n_steps - 1):
        for _ in range(ref_substeps):
            pos, vel, acc = leapfrog_step(pos, vel, acc, mass, dt_fine,
                                          epsilon, g=g)
            step_idx += 1
        if (k + 1) % sample_every == 0:
            pos_list.append(pos.copy())
            vel_list.append(vel.copy())
            e_list.append(total_energy(pos, vel, mass, epsilon, g=g))
            t_list.append((k + 1) * dt_N)

    return {
        "pos":    np.stack(pos_list),
        "vel":    np.stack(vel_list),
        "energy": np.asarray(e_list),
        "t":      np.asarray(t_list),
    }


def reference_kepler(primary_mass: float,
                      secondary_mass: float,
                      r_vec0: np.ndarray,
                      v_vec0: np.ndarray,
                      t: np.ndarray,
                      g: float = DEFAULT_GRAVITY_G) -> dict:
    """
    Closed-form 2-body Keplerian orbit propagated from the caller's
    actual relative state at t=0 (primary at origin).

    Returns the same dict shape as `reference_leapfrog` so the runner
    is reference-agnostic. Useful for the Sun-Earth-only preset, and
    as the "book" orbit generator in `real_case_runner`.

    The orbit's semi-major axis, eccentricity AND the initial phase /
    orbital-plane orientation are all derived from the initial state
    (r_vec0, v_vec0), so the propagated orbit passes through the IC at
    t = 0 exactly — not through an arbitrary periapsis-passage point
    on a fixed in-plane axis.

    Parameters
    ----------
    primary_mass   : float, M1 in the same units as `secondary_mass`
    secondary_mass : float, M2
    r_vec0         : (3,) : secondary-minus-primary position at t=0
                             (same length unit as the caller's system;
                             dimensionless here)
    v_vec0         : (3,) : secondary-minus-primary velocity at t=0
    t              : (T,) : sample times (same time unit as the
                              caller's system; dimensionless here)
    g              : float, gravitational constant (= 1 in N-body
                              units; matches the rest of the runner)
    """
    r_vec0 = np.asarray(r_vec0, dtype=np.float64)
    v_vec0 = np.asarray(v_vec0, dtype=np.float64)
    mu = g * (primary_mass + secondary_mass)

    r0 = float(np.linalg.norm(r_vec0))
    v0 = float(np.linalg.norm(v_vec0))
    # Osculating elements straight from the relative state.
    a = 1.0 / (2.0 / r0 - v0 ** 2 / mu)
    if a <= 0:
        raise ValueError("non-positive semi-major axis in Kepler reference "
                         "(hyperbolic input state)")
    rv_dot = float(np.dot(r_vec0, v_vec0))
    e_vec = ((v0 ** 2 - mu / r0) * r_vec0 - rv_dot * v_vec0) / mu
    e = float(np.linalg.norm(e_vec))
    if e < 1e-12:
        e = 0.0

    # Perifocal basis from the IC: p_hat points at periapsis (or at
    # the body itself for a perfect circle, where periapsis is
    # degenerate), w_hat along the angular momentum, q_hat completes
    # the right-handed frame.
    if e > 0.0:
        p_hat = e_vec / e
    else:
        p_hat = r_vec0 / r0
    h_vec = np.cross(r_vec0, v_vec0)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag < 1e-30:
        raise ValueError("degenerate (purely radial) Kepler input state")
    w_hat = h_vec / h_mag
    q_hat = np.cross(w_hat, p_hat)

    # Initial anomalies: nu0 from the basis, M0 from E0 so the closed
    # form is anchored to the IC instead of to periapsis passage.
    nu0 = math.atan2(float(np.dot(r_vec0, q_hat)), float(np.dot(r_vec0, p_hat)))
    E0 = math.atan2(math.sqrt(max(1.0 - e * e, 0.0)) * math.sin(nu0),
                    e + math.cos(nu0))
    M0 = E0 - e * math.sin(E0)
    n_mean = math.sqrt(mu / a ** 3)

    pos_list, vel_list, e_list = [], [], []
    for ti in t:
        # Solve Kepler's equation M = E - e sin E.
        M = M0 + n_mean * ti
        E = M
        for _ in range(10):
            E = E - (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                              math.sqrt(1.0 - e)  * math.cos(E / 2.0))
        r  = a * (1.0 - e * math.cos(E))

        # Secondary in perifocal frame (primary at foci).
        x = r * math.cos(nu)
        y = r * math.sin(nu)
        vx = -mu * math.sin(nu) / h_mag
        vy =  mu * (e + math.cos(nu)) / h_mag

        # Rotate back into the global frame (keeps inclination).
        pos_s = x * p_hat + y * q_hat
        vel_s = vx * p_hat + vy * q_hat

        # Primary is stationary at origin in this frame, but to match
        # the convention (pos, vel for *both* bodies) we compute the
        # primary's state from the COM.
        M_total = primary_mass + secondary_mass
        pos_p = -(secondary_mass / M_total) * pos_s
        vel_p = -(secondary_mass / M_total) * vel_s

        pos_pair = np.stack([pos_p, pos_s], axis=0)
        vel_pair = np.stack([vel_p, vel_s], axis=0)
        pos_list.append(pos_pair)
        vel_list.append(vel_pair)
        e_list.append(0.5 * (primary_mass * np.dot(vel_p, vel_p)
                             + secondary_mass * np.dot(vel_s, vel_s))
                      - g * primary_mass * secondary_mass / r)

    return {
        "pos":    np.stack(pos_list),
        "vel":    np.stack(vel_list),
        "energy": np.asarray(e_list),
        "t":      np.asarray(t),
    }
