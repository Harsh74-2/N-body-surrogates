"""
unit_rescale.py
===============
Rescale a real (SI / AU) N-body system into the dimensionless
N-body units used by `simulation_3d` and the trained surrogates.

The training distribution is fully characterised by:
    G       = 1
    Σ mass  = 1
    Length  = 1
    Time    = 1       (the natural unit T = sqrt(L³ / (G·M)))

A real system with total mass M and characteristic length L is mapped
onto this convention by dividing all positions by L, all velocities by
sqrt(G·M/L), all masses by M, all times by T = sqrt(L³/(G·M)), and
all energies by (G·M²/L).

The choices of M and L are auditable: they are reported in the
per-preset `summary.json` along with the inverse scaling back to SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ── Physical constants (SI) ───────────────────────────────────────────────────
# Using SI for portability; conversions to AU happen *inside* the
# presets, so this module is unit-agnostic, it just rescales by the
# scale factors the caller provides.
from pipeline_config import AU_M, DAY_S, G_SI  # SI base units


@dataclass
class UnitScale:
    """The rescaling factors and their inverses. All in SI."""
    M: float                # mass unit (kg)
    L: float                # length unit (m)
    T: float                # time unit (s): derived

    @property
    def V(self) -> float:
        """Velocity unit (m/s) = L / T."""
        return self.L / self.T

    @property
    def E(self) -> float:
        """Energy unit (J) = G·M²/L."""
        return G_SI * self.M ** 2 / self.L

    def to_dict(self) -> dict:
        return {
            "M_kg":      self.M,
            "L_m":       self.L,
            "T_s":       self.T,
            "V_m_per_s": self.V,
            "E_J":       self.E,
        }


def make_scale(M: float, L: float) -> UnitScale:
    """
    Build a UnitScale from a mass unit and a length unit.

    Time unit derived: T = sqrt(L³ / (G·M)).
    """
    T = math.sqrt(L ** 3 / (G_SI * M))
    return UnitScale(M=M, L=L, T=T)


def scale_for_preset(bodies: list[dict], preset: dict) -> UnitScale:
    """
    Build a UnitScale for a real preset.

    Mass unit M = total system mass in kg (so Σ mass_N = 1 exactly,
    matching the trained distribution).

    Length unit L = outermost semi-major axis in metres, or the
    `characteristic_radius_au` override if the preset supplies one.

    For the disc-IMF baseline we use the disc scale radius (R_d ≈ 1
    in N-body units, see `simulation_3d.init_galaxy_disc`).
    """
    total_mass_kg = sum(b["mass_kg"] for b in bodies)

    # Length unit.
    override_au = preset.get("characteristic_radius_au")
    if override_au is not None:
        L_m = override_au * AU_M
    elif preset.get("in_distribution", False):
        # In the synthetic disc unit system, the natural length is the
        # disc scale radius R_d. simulation_3d places R_d at ~1 in its
        # dimensionless units; we pick a physical scale R_d = 5 kpc
        # (typical spiral).
        L_m = 5.0e3 * 3.0857e16
    else:
        # Outermost body distance from the origin. For the Solar-System
        # presets the Sun is at the origin so this is the outermost
        # planet's semi-major axis; for the planetocentric Galilean
        # preset Jupiter is at the origin so this is Callisto's orbit.
        # Presets may also pin this down with an explicit
        # `characteristic_radius_au` override (the Galilean preset does).
        L_m = _outermost_a_m(bodies, preset)

    return make_scale(M=total_mass_kg, L=L_m)


def _outermost_a_m(bodies: list[dict], preset: dict) -> float:
    """
    Pick a characteristic length in metres for a real preset: the
    distance of the outermost body from the origin.

    For the Solar-System presets the Sun sits at the origin, so this is
    the outermost planet's semi-major axis. For the planetocentric
    Galilean preset Jupiter sits at the origin, so this is Callisto's
    orbit. (The Galilean preset also sets an explicit
    `characteristic_radius_au` override, so this path is only a fallback
    there.)
    """
    pos = np.array([b["pos_au"] for b in bodies])    # (N, 3), in AU
    r_au = float(np.max(np.linalg.norm(pos, axis=1)))
    return r_au * AU_M


def rescale_ic(bodies: list[dict], scale: UnitScale) -> dict:
    """
    Rescale a list of body dicts (with pos_au, vel_au_per_day, mass_kg)
    into dimensionless N-body units.

    Returns
    -------
    dict with arrays:
        pos : (N, 3): dimensionless position
        vel : (N, 3): dimensionless velocity
        mass: (N,)  : dimensionless mass (Σ = 1 by construction)
        names: list[str]
    """
    pos_au = np.array([b["pos_au"] for b in bodies], dtype=np.float64)
    vel_au_per_day = np.array([b["vel_au_per_day"] for b in bodies],
                              dtype=np.float64)
    mass_kg = np.array([b["mass_kg"] for b in bodies], dtype=np.float64)

    # Convert astronomical units → SI → dimensionless.
    pos_m       = pos_au * AU_M
    vel_m_per_s = vel_au_per_day * AU_M / DAY_S
    pos_N       = pos_m / scale.L
    vel_N       = vel_m_per_s / scale.V
    mass_N      = mass_kg / scale.M

    # Strip the system's centre-of-mass velocity so the network sees
    # a zero-momentum state (it was trained on discs with 0 net P).
    p_total = (mass_N[:, None] * vel_N).sum(axis=0)
    vel_N  -= p_total / mass_N.sum()

    return {
        "pos":  pos_N,
        "vel":  vel_N,
        "mass": mass_N,
        "names": [b["name"] for b in bodies],
    }


def rescale_dt(dt_seconds: float, scale: UnitScale) -> float:
    """Rescale a SI time interval into dimensionless units."""
    return dt_seconds / scale.T


def rescale_t_back(t_N: np.ndarray, scale: UnitScale) -> np.ndarray:
    """Convert a dimensionless time axis back to seconds."""
    return t_N * scale.T


def rescale_pos_back(pos_N: np.ndarray, scale: UnitScale) -> np.ndarray:
    """Convert dimensionless positions back to metres."""
    return pos_N * scale.L


def rescale_energy_back(E_N: np.ndarray, scale: UnitScale) -> np.ndarray:
    """Convert dimensionless energies back to joules."""
    return E_N * scale.E
