"""
ic_loader.py
============
Load and validate initial conditions for the real-case validation
pipeline. Two sources are supported:

1. A built-in preset name (one of `presets.PRESETS[*].name`).
2. A user-supplied JSON file with the same schema as a preset:

       {
         "name":  "my_scenario",
         "label": "My custom scenario",
         "bodies": [
           {"name": "...", "mass_kg": 1.0e30,
            "pos_au": (x, y, z), "vel_au_per_day": (vx, vy, vz)},
           ...
         ],
         "duration_years":     10,
         "sample_per_year":    12,
         "reference":          "leapfrog"   (default)
       }

After loading, this module rescales the IC into N-body units and
returns a `RescaledIC` dataclass with everything the runner needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline_config import (
    IC_BASE_SEED,
    IC_M_MAX,
    IC_M_MIN,
)

# The in-distribution synthetic-disc baseline uses a fixed body count.
# This is a property of that specific preset, not a global default.
DISC_BASELINE_N_BODIES: int = 25

from . import presets as preset_mod
from . import unit_rescale


@dataclass
class RescaledIC:
    """A preset rescaled into dimensionless N-body units."""
    name:        str
    label:       str
    reference:   str
    scale:       unit_rescale.UnitScale
    pos:         np.ndarray            # (N, 3), dimensionless
    vel:         np.ndarray            # (N, 3)
    mass:        np.ndarray            # (N,)
    names:       list[str]
    duration_years: float
    sample_per_year: int
    in_distribution: bool
    characteristic_length_m: float     # = scale.L, for report


def load_preset(name: str) -> RescaledIC:
    """
    Load a built-in preset by name and return a rescaled IC.

    The in-distribution disc-IMF baseline is generated via
    `simulation_3d.init_galaxy_disc(N=25, m_min=0.1, m_max=50)` and
    then Σ-normalised to match the trained unit convention.
    """
    p = preset_mod.get_preset(name)

    if p.get("in_distribution", False):
        return _load_in_distribution_disc(p)

    return _rescale(p)


def load_custom(path: str) -> RescaledIC:
    """Load a user-supplied IC JSON file and return a rescaled IC."""
    pth = Path(path)
    if not pth.is_file():
        raise FileNotFoundError(f"IC file not found: {path!r}")
    with open(pth, "r", encoding="utf-8") as f:
        p = json.load(f)
    _validate(p)
    return _rescale(p)


# ── Internals ────────────────────────────────────────────────────────────────
def _validate(p: dict) -> None:
    """Raise if a custom IC dict is missing required keys."""
    for key in ("name", "label", "bodies", "duration_years", "sample_per_year"):
        if key not in p:
            raise ValueError(f"IC dict missing key {key!r}: {p}")
    for b in p["bodies"]:
        for k in ("name", "mass_kg", "pos_au", "vel_au_per_day"):
            if k not in b:
                raise ValueError(f"body missing key {k!r}: {b}")


def _rescale(p: dict) -> RescaledIC:
    """Rescale a real preset (with mass_kg, pos_au, vel_au_per_day) into
    dimensionless units. The `in_distribution` flag is False by default
    for custom ICs."""
    scale = unit_rescale.scale_for_preset(p["bodies"], p)
    r = unit_rescale.rescale_ic(p["bodies"], scale)
    return RescaledIC(
        name=p["name"],
        label=p.get("label", p["name"]),
        reference=p.get("reference", "leapfrog"),
        scale=scale,
        pos=r["pos"],
        vel=r["vel"],
        mass=r["mass"],
        names=r["names"],
        duration_years=p["duration_years"],
        sample_per_year=p["sample_per_year"],
        in_distribution=p.get("in_distribution", False),
        characteristic_length_m=scale.L,
    )


def _load_in_distribution_disc(p: dict) -> RescaledIC:
    """
    Build the in-distribution baseline directly from
    `simulation_3d.init_galaxy_disc(N=25, m_min=0.5, m_max=5)` -- the
    same body count and mass range the surrogates were trained on, so
    this preset is a genuine in-distribution sanity check rather than an
    OOD-on-mass-ratio test. (The coarse time step is matched to training
    via `sample_per_year` in the preset, not here.)

    We re-scale to Σ mass = 1 (matching the trained unit convention) and
    set the characteristic length to the simulation's R_d (which is
    ~1.0 in N-body units by construction, see `simulation_3d.py`).
    """
    from simulation_3d import init_galaxy_disc

    pos, vel, mass = init_galaxy_disc(
        N=DISC_BASELINE_N_BODIES, seed=IC_BASE_SEED,
        m_min=IC_M_MIN, m_max=IC_M_MAX,
    )
    # init_galaxy_disc already returns Σmass = 1; just sanity-check.
    assert abs(mass.sum() - 1.0) < 1e-9, f"disc mass sum {mass.sum()}"
    # The disc is already in N-body units; characteristic length R_d = 1.
    scale = unit_rescale.make_scale(M=1.0, L=1.0)
    return RescaledIC(
        name=p["name"],
        label=p.get("label", p["name"]),
        reference=p.get("reference", "leapfrog"),
        scale=scale,
        pos=pos,
        vel=vel,
        mass=mass,
        names=[f"body_{i}" for i in range(len(mass))],
        duration_years=p["duration_years"],
        sample_per_year=p["sample_per_year"],
        in_distribution=True,
        characteristic_length_m=1.0,
    )
