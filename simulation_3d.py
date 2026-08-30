"""
simulation_3d.py
================
Generates 3D N-body gravitational trajectory datasets for neural network
training.

Default initial conditions are a galaxy disc in centrifugal equilibrium:
a Plummer core (finite central density, analytic M(<r)) smoothly blended
with an exponential-disc surface density (Freeman 1970, the real
galaxy profile), with bodies placed on exact circular orbits at each
radius. The disc is thin (Gaussian z scatter with σ_z ≪ R_d) and
rotates prograde. No transient settling phase, the integrator's job
is just to maintain the equilibrium, which keeps conservation tests
clean.

Outputs a single raw trajectory per simulation as a `.npz` archive:
    - frames  : ndarray, shape [Frames, N, 6]  (x, y, z, vx, vy, vz)
    - mass    : ndarray, shape [N,]            (body masses, normalised)
    - meta    : ndarray, shape (4,)            (dt, epsilon, G, seed)

The downstream pipeline (`3d_export_pipeline.py`) consumes `frames` only;
`mass` and `meta` are stored alongside so consumers (validation,
dashboards) can recover the physics without re-running the integrator.

Physics
-------
- Symplectic leapfrog (Störmer–Verlet) integrator, conserves energy over
  long runs, which matters for clean training labels.
- Direct O(N²) pairwise summation with Plummer-style softening
  (r² → r² + ε²) to avoid singularities at close encounters.
- Vectorised accelerations via broadcasting (no per-body Python loop).

Usage
-----
    from simulation_3d import generate_dataset
    generate_dataset(num_simulations=5, N=10, frames=5000)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import configure_utf8_stdout

configure_utf8_stdout()


from pipeline_config import (
    G,
    SOFTENING,
    DT,
    DEFAULT_N_FRAMES,
    DEFAULT_N_SIMULATIONS,
    GNN_NUM_SIMULATIONS,
    LSTM_NUM_SIMULATIONS,
    MLP_NUM_SIMULATIONS,
    ModelType,
    IC_R_CORE,
    IC_R_DISC,
    IC_R_MAX,
    IC_SIGMA_Z,
    IC_M_MIN,
    IC_M_MAX,
    IC_BASE_SEED,
)


# Re-export physics constants so existing imports keep working.
__all__ = ["generate_dataset", "compute_accelerations", "leapfrog_step",
           "total_energy", "init_galaxy_disc", "SimConfig",
           "G", "SOFTENING", "DT"]


# ── Container for a single simulation's metadata ─────────────────────────────
@dataclass(frozen=True)
class SimConfig:
    """Static configuration of one simulation, saved alongside the data."""
    n_bodies:   int
    n_frames:   int
    dt:         float
    epsilon:    float
    g:          float
    seed:       int


# ── Physics ──────────────────────────────────────────────────────────────────
def compute_accelerations(pos: np.ndarray,
                          mass: np.ndarray,
                          epsilon: float,
                          g: float = G) -> np.ndarray:
    """
    Gravitational acceleration on every body via pairwise O(N²) summation.

    Uses broadcasting, the loop is moved into NumPy. Self-interaction is
    explicitly zeroed so a body's own mass does not act on itself.

    Parameters
    ----------
    pos     : ndarray, shape (N, 3)
    mass    : ndarray, shape (N,)
    epsilon : float, softening length
    g       : float, gravitational constant

    Returns
    -------
    acc : ndarray, shape (N, 3)
    """
    # diff[i, j] = pos[j] - pos[i]   → shape (N, N, 3)
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]

    # Softened squared distances: r² + ε²
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff) + epsilon ** 2

    # Force magnitude per (i, j): g * m_j / r³
    # r³ = (r²)^(3/2); avoid 0**negative by softening (ε > 0 guarantees finite r²)
    inv_r_cube = 1.0 / (dist_sq ** 1.5)

    # Zero self-interaction (i == j)
    np.fill_diagonal(inv_r_cube, 0.0)

    # acc[i] = Σ_j  g * m_j * (pos[j] - pos[i]) / r³
    # factor[i, j] = g * m[j] * inv_r_cube[i, j]
    factor = g * mass[np.newaxis, :] * inv_r_cube          # shape (N, N)
    acc    = np.einsum("ij,ijk->ik", factor, diff)         # shape (N, 3)
    return acc


def leapfrog_step(pos: np.ndarray,
                  vel: np.ndarray,
                  acc: np.ndarray,
                  mass: np.ndarray,
                  dt: float,
                  epsilon: float,
                  g: float = G) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One leapfrog (Störmer–Verlet) update.

        v(t+½dt) = v(t)  + ½ a(t)  dt
        x(t+dt)  = x(t)  +  v(t+½dt) dt
        a(t+dt)  = f(x(t+dt))
        v(t+dt)  = v(t+½dt) + ½ a(t+dt) dt
    """
    vel_half = vel + 0.5 * acc * dt
    pos_new  = pos + vel_half * dt
    acc_new  = compute_accelerations(pos_new, mass, epsilon, g=g)
    vel_new  = vel_half + 0.5 * acc_new * dt
    return pos_new, vel_new, acc_new


# ── Diagnostics ──────────────────────────────────────────────────────────────
def total_energy(pos: np.ndarray,
                 vel: np.ndarray,
                 mass: np.ndarray,
                 epsilon: float,
                 g: float = G) -> float:
    """
    Total mechanical energy E = KE + PE.

        KE = ½ Σᵢ mᵢ |vᵢ|²
        PE = − Σ_{i<j} g mᵢ mⱼ / √(r²ᵢⱼ + ε²)

    The upper-triangular loop is O(N²/2) but only used for validation,
    not inside the hot path.
    """
    ke = 0.5 * np.sum(mass * np.sum(vel ** 2, axis=1))

    pe = 0.0
    for i in range(pos.shape[0]):
        diff = pos[i + 1:] - pos[i]
        dist = np.sqrt(np.sum(diff ** 2, axis=1) + epsilon ** 2)
        pe  -= g * mass[i] * np.sum(mass[i + 1:] / dist)

    return float(ke + pe)


# ── Initial conditions ───────────────────────────────────────────────────────
def init_galaxy_disc(N: int,
                     r_core: float = 1.0,
                     r_disc: float = 1.5,
                     r_max: float = 6.0,
                     sigma_z: float = 0.075,
                     m_min: float = 0.5,
                     m_max: float = 5.0,
                     g: float = G,
                     seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stable 3D galaxy-disc initial conditions in centrifugal equilibrium.

    The surface density blends a Plummer core (finite central density,
    analytic M(<r) for r < r_core) with an exponential disc
    (Σ(R) ∝ exp(−R/r_disc), the Freeman 1970 profile of real spiral
    galaxies) for R > r_core. The blend is continuous at r_core because
    the Plummer profile is already vanishingly small there.

    Each body is placed on an exact circular orbit at its cylindrical
    radius R, with tangential speed v_φ = √(G·M(<R)/R). This puts the
    disc in perfect centrifugal equilibrium from t=0, no transient
    settling phase, so the integrator's energy drift is just the
    integrator's, not relaxation physics. v_φ > 0 (prograde) by
    construction.

    Parameters
    ----------
    N        : number of bodies
    r_core   : Plummer core radius (also the inner blend radius)
    r_disc   : exponential-disc scale length
    r_max    : outer cutoff for the radius grid (≈ 4·r_disc)
    sigma_z  : Gaussian rms of the z scatter, thin-disc thickness
    m_min    : minimum body mass (log-uniform sampling). Pass real-MSun
               values (e.g. 0.1) if you want solar-mass ranges before the
               Σ=1 normalisation; the network learns mass *ratios* and
               these are preserved by renormalisation.
    m_max    : maximum body mass (log-uniform sampling). Same convention
               as `m_min`: typical stellar upper bound ≈ 50–100 M☉.
    g        : gravitational constant (matches module default)
    seed     : RNG seed

    Returns
    -------
    pos  : (N, 3) float64 , (x, y, z); x-y plane is the disc, z is thickness
    vel  : (N, 3) float64 , (vx, vy, vz); v_φ in x-y plane, vz ≈ 0
    mass : (N,)   float64 , Σmass = 1
    """
    rng = np.random.default_rng(seed)

    # ── Build the radial CDF once: Σ(R) ∝ exp(−R/r_disc), R ∈ [0, r_max] ─
    # We use surface density (not volume) because the disc is thin; the
    # body's "radius" for the orbit is the cylindrical radius R.
    n_grid = 1000
    R_grid = np.linspace(0.0, r_max, n_grid + 1)
    dR     = R_grid[1] - R_grid[0]
    # Plummer core weighting: smoothly turns on at R=0, dies off by R=r_core.
    # This is a multiplicative weight on the exponential, at small R it
    # suppresses the exponential's exp(0)=1 peak and replaces it with the
    # well-behaved Plummer core; at R > r_core it's essentially zero, so
    # the exponential dominates the outer disc.
    plummer_weight = (1.0 + (R_grid / r_core) ** 2) ** (-2.5)
    sigma_R        = np.exp(-R_grid / r_disc) * plummer_weight * R_grid  # 2πR dR factor
    cdf            = np.cumsum(sigma_R) * dR
    cdf           /= cdf[-1]                    # normalise to [0, 1]
    # Guard against the (theoretical) CDF reaching exactly 1.0 at the
    # last grid point: clip so the inverse interpolation is well-defined.
    cdf[-1] = 1.0

    # ── Sample R by inverse-CDF ─────────────────────────────────────────────
    u = rng.uniform(0.0, 1.0, size=N)
    R = np.interp(u, cdf, R_grid)

    # ── Azimuthal angle φ uniform on [0, 2π) ───────────────────────────────
    phi = rng.uniform(0.0, 2.0 * np.pi, size=N)

    # ── Cartesian positions in x-y plane + small Gaussian z scatter ────────
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = rng.normal(0.0, sigma_z, size=N)
    pos = np.stack([x, y, z], axis=1)

    # ── Build the cumulative-mass table M(<R) for v_φ = √(G·M/R) ───────────
    # Mass enclosed inside cylindrical radius R, computed by summing the
    # surface density × 2πR dR up to R. We bin the bodies-on-grid test
    # mass evenly over the grid: Σ(mass per area) × 2πR dR per shell.
    M_grid = np.cumsum(sigma_R) * dR          # ∝ enclosed mass per unit total mass
    M_grid /= M_grid[-1]                      # normalise so M(<r_max) = 1
    # Guard at the upper end so interpolation is defined.
    M_grid[-1] = 1.0
    # Note: this is a *mass per unit total-mass* curve; the actual enclosed
    # mass is M_grid × Σmass = M_grid × 1 = M_grid (because we normalise
    # the body masses to Σ=1 below).
    M_of_R = np.interp(R, R_grid, M_grid)

    # ── Circular orbital speed at each body's radius ───────────────────────
    # v_φ = √(G · M(<R) / R). Guard R=0 (vanishingly rare with rejection
    # at the IC level; the interpolation grid starts at R=0 already so
    # M(0)=0 gives v_φ=0 for any body that lands at R=0).
    v_phi = np.sqrt(np.clip(g * M_of_R / np.maximum(R, 1e-12), 0.0, None))

    # ── Velocities: tangential in x-y plane, no radial, no z motion ───────
    # Tangential unit vector is (−sin φ, cos φ, 0).
    vel = np.stack([-v_phi * np.sin(phi),
                     v_phi * np.cos(phi),
                     np.zeros_like(v_phi)], axis=1)

    # ── Masses: log-uniform, normalised to Σ = 1 ─────────────────────────────
    mass = np.exp(rng.uniform(np.log(m_min), np.log(m_max), N))
    mass /= mass.sum()

    return pos, vel, mass


# ── Dataset generator ────────────────────────────────────────────────────────
def generate_dataset(num_simulations: int,
                     N: int,
                     frames: int,
                     dt: float = DT,
                     epsilon: float = SOFTENING,
                     g: float = G,
                     output_dir: str | None = None,
                     base_seed: int = 42,
                     r_core: float = 1.0,
                     r_disc: float = 1.5,
                     sigma_z: float = 0.075,
                     m_min: float = 0.5,
                     m_max: float = 5.0,
                     dtype: np.dtype = np.float32) -> list[str]:
    """
    Generate `num_simulations` independent N-body trajectories and save each
    as a `.npz` archive in `output_dir`.

    File layout (per simulation):
        sim_N{N}_{idx:03d}.npz
            frames : (frames, N, 6) float32, (x, y, z, vx, vy, vz) per body
            mass   : (N,)            float64, body masses, Σ = 1
            meta   : (4,)            float64, (dt, epsilon, G, seed)

    Parameters
    ----------
    num_simulations : number of independent trajectories
    N               : number of bodies per simulation
    frames          : number of leapfrog steps per trajectory
    dt              : integration time step
    epsilon         : Plummer softening length
    g               : gravitational constant
    output_dir      : directory for output files (created if missing).
                      If None (default), resolves to `<project>/raw_data/`
                      using `__file__` so the dataset lands next to the
                      script regardless of cwd. Pass an absolute or
                      relative path to override.
    base_seed       : master RNG seed; per-simulation seed = base_seed + idx
    r_core          : Plummer core radius / inner blend radius
    r_disc          : exponential-disc scale length
    sigma_z         : Gaussian rms of z scatter (thin-disc thickness)
    m_min           : minimum body mass for log-uniform sampling.
                      Pass real-MSun values (e.g. 0.1) to mimic a stellar IMF
                      before the Σ=1 normalisation below. Default (0.5, 5.0)
                      reproduces the legacy IMF-style range.
    m_max           : maximum body mass for log-uniform sampling.
                      Real-MSun upper bound ≈ 50–100.
    dtype           : storage dtype for the frames array (float32 default)

    Returns
    -------
    paths : list of written file paths
    """
    out_path = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "raw_data"

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Generating {num_simulations} simulations | N={N} | frames={frames} "
          f"| dt={dt} | eps={epsilon} | G={g} | init=disc")

    paths: list[str] = []
    for sim_idx in range(num_simulations):
        start = time.perf_counter()
        seed  = base_seed + sim_idx

        pos, vel, mass = init_galaxy_disc(
            N, r_core=r_core, r_disc=r_disc, sigma_z=sigma_z,
            g=g, seed=seed, m_min=m_min, m_max=m_max,
        )
        acc = compute_accelerations(pos, mass, epsilon, g=g)

        # Frame buffer: (frames, N, 6): stored as float32 for compactness
        trajectory = np.zeros((frames, N, 6), dtype=dtype)

        # Initial energy (for drift diagnostics)
        e0 = total_energy(pos, vel, mass, epsilon, g=g)

        for f in range(frames):
            trajectory[f, :, 0:3] = pos.astype(dtype, copy=False)
            trajectory[f, :, 3:6] = vel.astype(dtype, copy=False)
            pos, vel, acc = leapfrog_step(pos, vel, acc, mass, dt, epsilon, g=g)

        # Final energy, printed for sanity; downstream code does not consume it
        e1 = total_energy(pos, vel, mass, epsilon, g=g)
        drift = (e1 - e0) / abs(e0) if e0 != 0 else float("nan")

        file_path = out_path / f"sim_N{N}_{sim_idx:03d}.npz"
        np.savez_compressed(
            file_path,
            frames=trajectory,
            mass=mass.astype(np.float64),
            meta=np.array([dt, epsilon, g, seed], dtype=np.float64),
        )
        paths.append(str(file_path))

        elapsed = time.perf_counter() - start
        print(f"  [{sim_idx + 1}/{num_simulations}] {file_path} "
              f"| {elapsed:5.2f}s | ΔE/E₀ = {drift:+.2e} | seed={seed}")

    return paths


# ── CLI entry point ──────────────────────────────────────────────────────────
def _default_num_sims(model_type: str | None) -> int:
    """Return the recommended number of simulations for a given model type."""
    if model_type == ModelType.MLP:
        return MLP_NUM_SIMULATIONS
    if model_type == ModelType.LSTM:
        return LSTM_NUM_SIMULATIONS
    if model_type == ModelType.GNN:
        return GNN_NUM_SIMULATIONS
    return DEFAULT_N_SIMULATIONS


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    CLI for `generate_dataset`. N is required; --num-simulations defaults
    to a per-model recommendation unless --model-type is also omitted, in
    which case the generic DEFAULT_N_SIMULATIONS is used.
    """
    p = argparse.ArgumentParser(
        description="Generate 3D N-body trajectories (.npz) for ML training.",
    )
    p.add_argument("--num-simulations", type=int,   default=None,
                   help="Number of independent trajectories to generate. "
                        "Defaults to a per-model recommendation when --model-type is set.")
    p.add_argument("--N",              type=int,   required=True,
                   help="Number of bodies per simulation (required, N is an experimental variable).")
    p.add_argument("--model-type",     default=None,
                   choices=list(ModelType.values()),
                   help="Target downstream model (mlp/lstm/gnn). Sets the "
                        "default --num-simulations to the recommended value for that architecture.")
    p.add_argument("--frames",         type=int,   default=DEFAULT_N_FRAMES,
                   help="Number of leapfrog steps per trajectory.")
    p.add_argument("--dt",             type=float, default=DT,
                   help="Integration time step (N-body units).")
    p.add_argument("--epsilon",        type=float, default=SOFTENING,
                   help="Plummer softening length.")
    p.add_argument("--g",              type=float, default=G,
                   help="Gravitational constant (N-body units).")
    p.add_argument("--output-dir",     type=str,   default=None,
                   help="Directory for sim_*.npz files. Default: "
                        "<project>/raw_data/ (resolved via __file__, so "
                        "it always lands next to this script regardless "
                        "of cwd).")
    p.add_argument("--base-seed",      type=int,   default=IC_BASE_SEED,
                   help="Master RNG seed; per-sim seed = base_seed + idx.")
    p.add_argument("--r-core",         type=float, default=IC_R_CORE,
                   help="Plummer core / inner blend radius.")
    p.add_argument("--r-disc",         type=float, default=IC_R_DISC,
                   help="Exponential-disc scale length.")
    p.add_argument("--sigma-z",        type=float, default=IC_SIGMA_Z,
                   help="Gaussian rms of z scatter.")
    p.add_argument("--m-min",          type=float, default=IC_M_MIN,
                   help="Minimum body mass (real-MSun values like 0.1 "
                        "give a stellar IMF before Σ=1 normalisation).")
    p.add_argument("--m-max",          type=float, default=IC_M_MAX,
                   help="Maximum body mass (real-MSun upper bound "
                        "≈ 50–100).")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    num_sims = args.num_simulations if args.num_simulations is not None else _default_num_sims(args.model_type)
    generate_dataset(
        num_simulations=num_sims,
        N=args.N,
        frames=args.frames,
        dt=args.dt,
        epsilon=args.epsilon,
        g=args.g,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        r_core=args.r_core,
        r_disc=args.r_disc,
        sigma_z=args.sigma_z,
        m_min=args.m_min,
        m_max=args.m_max,
    )