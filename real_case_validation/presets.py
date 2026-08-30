"""
presets.py
==========
Real-Solar-System initial conditions for the real-case validation
pipeline. Positions and velocities are heliocentric J2000 ecliptic,
expressed in AU and AU/day (the canonical NASA Horizons frame).

Sources
-------
- Initial states (positions and velocities): NASA JPL Horizons system,
  geometric ecliptic-of-J2000.0 state vectors at epoch 2026-08-07
  00:00 TDB (JD 2461259.5), Sun-centred (DE441 ephemeris); the Moon is
  taken geocentrically and added to Earth's heliocentric state.
  Retrieved 2026-08-07 from https://ssd.jpl.nasa.gov/horizons/ .
- Masses: NASA Planetary Fact Sheet (Williams 2024,
  https://nssdc.gsfc.nasa.gov/planetary/factsheet/, retrieved 2026-07).
- The Keplerian element tables (`_PLANETS`, `_DWARF_PLANETS`) that were
  retained "for reference" in earlier revisions have been removed; the
  canonical initial states come from the Horizons 2026-08-07 vectors in
  `_HORIZONS_2026` (planets) and `_DWARF_HORIZONS_2026` (dwarfs), which
  are more accurate than advancing J2000 mean elements by mean motion.
- Galilean moons: real JUP230 ephemerides are out of scope, so the
  `jupiter_galileans` preset and the Galilean components of
  `solar_system_extended` use self-consistent toy circular orbits
  (real masses and orbital radii) around Jupiter's Horizons-2026
  position. This is documented as a limitation.

Schema
------
Each preset is a dict with:
    name           : unique slug for filenames
    label          : human-readable label
    bodies         : list of {name, mass_kg, pos_au, vel_au_per_day}
    duration_years : total integration window
    sample_per_year: how many samples to record per year
    reference      : "leapfrog" (default) or "kepler" (2-body only)
    characteristic_radius_au
                   : optional override for the rescaling length unit;
                     default = outermost semi-major axis.
"""

from __future__ import annotations

import math

from pipeline_config import AU_M, DAY_S, JUPITER_MASS_KG, SUN_MASS_KG

# Epoch of the real-case initial conditions: NASA JPL Horizons geometric
# ecliptic-of-J2000.0 state vectors, retrieved 2026-08-07.
HORIZONS_EPOCH = "2026-08-07 00:00 TDB (JD 2461259.5)"

# ── Solar System constants (NASA fact sheet) ──────────────────────────────────
GM_SUN_AU3_DAY2 = 2.959122082855911e-4   # G·M_sun in AU³/day² (Standish 1992)
_KM_TO_AU = 1000.0 / AU_M            # km -> AU (AU_M is metres per AU; 1 km = 1000 m)
_S_TO_DAY = 1.0 / DAY_S


# ── Horizons state vectors, epoch 2026-08-07 00:00 TDB (JD 2461259.5) ────────
# Geometric ecliptic-of-J2000.0 heliocentric states (AU, AU/day), DE441.
# Retrieved 2026-08-07 from NASA JPL Horizons (ssd.jpl.nasa.gov/horizons).
# These are the canonical initial states (masses and state vectors).
# Format: name -> (mass_kg, (x, y, z)_AU, (vx, vy, vz)_AU/day)
_HORIZONS_2026 = {
    "mercury": (3.3011e23,
               ( 2.733586046168486e-01,  1.732858282375765e-01, -1.090992826995499e-02),
               (-2.057146881794168e-02,  2.498928539013271e-02,  3.928966975320862e-03)),
    "venus":   (4.8675e24,
               (-4.547541389113513e-02, -7.253145485887759e-01, -7.341116375947924e-03),
               ( 2.005079101173271e-02, -1.342452262606902e-03, -1.175356148581843e-03)),
    "earth":   (5.9722e24,
               ( 7.065849396576847e-01, -7.275354787072639e-01,  3.672761075317352e-05),
               ( 1.206876457688077e-02,  1.191837756930655e-02, -8.034460940694388e-07)),
    "mars":    (6.4171e23,
               ( 8.189535970266916e-01,  1.241595067058806e+00,  5.938237568504488e-03),
               (-1.114949001008605e-02,  8.897086137635154e-03,  4.598439135452572e-04)),
    "jupiter": (1.8982e27,
               (-3.162566119148378e+00,  4.238670634428467e+00,  5.315058168550223e-02),
               (-6.139974225167400e-03, -4.163974389350911e-03,  1.547165657779963e-04)),
    "saturn":  (5.6832e26,
               ( 9.328749084785100e+00,  1.465184127207989e+00, -3.968580131809155e-01),
               (-1.176013643987789e-03,  5.497974121885549e-03, -4.917313039828926e-05)),
    "uranus":  (8.6810e25,
               ( 9.124256935024416e+00,  1.717831258735838e+01, -5.450856888989106e-02),
               (-3.508570506756798e-03,  1.659688171750842e-03,  5.174137611677626e-05)),
    "neptune": (1.0241e26,
               ( 2.984649099414190e+01,  1.206764408457145e+00, -7.126512575118024e-01),
               (-1.535543296982131e-04,  3.152698515317247e-03, -6.150317214546341e-05)),
}

# The Sun is anchored at the origin (heliocentric frame). The runner
# subtracts the system COM velocity so the network sees a zero-momentum
# state, matching the training distribution.
_SUN = {
    "name": "Sun",
    "mass_kg": SUN_MASS_KG,
    "pos_au": (0.0, 0.0, 0.0),
    "vel_au_per_day": (0.0, 0.0, 0.0),
}


# ── Shared geocentric Moon state (Horizons, 2026-08-07) ──────────────────────
# Used by both `sun_planets_moon` and `solar_system_extended` presets.
# Geocentric ecliptic-of-J2000.0 state vector (AU, AU/day) at the same
# Horizons epoch as the planets; added to Earth's heliocentric state to
# give the Moon's heliocentric state. Retrieved 2026-08-07.
_MOON_GEO_POS_AU       = ( 1.378508143046396e-03,  2.042034630309507e-03,  2.264047959648789e-04)
_MOON_GEO_VEL_AU_DAY   = (-5.190754480579388e-04,  3.227863485338149e-04,  1.360297202101127e-06)
_MOON_MASS_KG          = 7.342e22


def _moon_relative_to_earth(earth: dict) -> dict:
    """Return the Moon's heliocentric state given Earth's heliocentric state.

    The geocentric offset is already in AU / AU-per-day (Horizons 2026-08-07),
    so it is added directly to Earth's heliocentric state.
    """
    return {
        "name": "Moon",
        "mass_kg": _MOON_MASS_KG,
        "pos_au": tuple(earth["pos_au"][i] + _MOON_GEO_POS_AU[i] for i in range(3)),
        "vel_au_per_day": tuple(
            earth["vel_au_per_day"][i] + _MOON_GEO_VEL_AU_DAY[i] for i in range(3)
        ),
    }


def _planet(name: str) -> dict:
    """Return the planet's Horizons-2026 heliocentric state as a body dict.

    Uses the geometric ecliptic-of-J2000.0 state vector retrieved from
    NASA JPL Horizons at epoch 2026-08-07 00:00 TDB (JD 2461259.5).
    """
    mass_kg, pos_au, vel_au_per_day = _HORIZONS_2026[name]
    return {
        "name": name.capitalize(),
        "mass_kg": mass_kg,
        "pos_au": tuple(pos_au),
        "vel_au_per_day": tuple(vel_au_per_day),
    }


def _preset(name: str, label: str, planet_names: list[str],
            duration_years: float, sample_per_year: int,
            reference: str = "leapfrog") -> dict:
    return {
        "name": name,
        "label": label,
        "bodies": [_SUN] + [_planet(n) for n in planet_names],
        "duration_years": duration_years,
        "sample_per_year": sample_per_year,
        "reference": reference,
    }


# ── The Galilean-moon preset (toy circular orbits) ───────────────────────────
# Real Galilean moon ephemerides (JUP230 etc.) are out of scope; we
# generate a *self-consistent* circular-orbit initial state with the
# real masses and J2000 orbital radii, which is the kind of toy
# configuration the surrogates might see in their training distribution
# (one heavy primary + 4 light satellites). This is the most honest way
# to validate "small-N bound clusters" without an external ephemeris.
_GALILEAN_TOY = [
    # name,       mass_kg,    a_km
    ("Io",         8.9319e22,  421_800.0),
    ("Europa",     4.7998e22,  671_034.0),
    ("Ganymede",   1.4819e23,  1_070_412.0),
    ("Callisto",   1.0759e23,  1_882_709.0),
]
_JUPITER_AU   = 5.20288700
_GM_JUPITER   = GM_SUN_AU3_DAY2 * (JUPITER_MASS_KG / SUN_MASS_KG)


def _galilean_moons() -> dict:
    # Planetocentric toy: Jupiter at the origin, the 4 Galilean moons on
    # circular orbits around it. We drop the Sun and Jupiter's heliocentric
    # motion on purpose: the thing we want to validate is the satellite
    # subsystem, and scaling it by the largest satellite orbit (Callisto)
    # keeps every body at O(1) in N-body units. Scaling by Jupiter's
    # heliocentric radius instead crushes the moons to ~1e-3 and makes the
    # system unresolvable for the surrogate, which is the bug this fixes.
    bodies = [{
        "name": "Jupiter",
        "mass_kg": JUPITER_MASS_KG,
        "pos_au": (0.0, 0.0, 0.0),
        "vel_au_per_day": (0.0, 0.0, 0.0),
    }]
    for name, mass, a_km in _GALILEAN_TOY:
        a_au = a_km * _KM_TO_AU
        # Circular speed around Jupiter only (the Sun is not in this system).
        v_au_per_day = math.sqrt(_GM_JUPITER / a_au)
        bodies.append({
            "name": name,
            "mass_kg": mass,
            "pos_au": (a_au, 0.0, 0.0),
            "vel_au_per_day": (0.0, v_au_per_day, 0.0),
        })
    # Characteristic length = Callisto's orbit (largest satellite a), so the
    # moons sit at O(1) in N-body units. With M ~ Jupiter mass the time unit
    # is ~2.65 days, so 4 samples/day gives dt_N ~ 0.09 (inside the training
    # range) and Io (1.769-day period) is sampled at ~7 points per orbit.
    callisto_a_au = 1_882_709.0 * _KM_TO_AU
    return {
        "name": "jupiter_galileans",
        "label": "Jupiter + 4 Galilean moons (toy circular orbits)",
        "bodies": bodies,
        "characteristic_radius_au": callisto_a_au,
        "duration_years": 1,
        "sample_per_year": 1460,       # 4 samples/day; Io orbit ~ 1.769 days -> ~7 samples/orbit
        "reference": "leapfrog",
    }


# ── Dwarf-planet Horizons state vectors (epoch 2026-08-07) ───────────────────
# These replace the J2000 Keplerian element table (`_DWARF_PLANETS`) that was
# previously retained "for reference" but indexed nowhere in the active code.
# Masses come straight from the Horizons table; the full state vectors do too.
# Format: name -> (mass_kg, (x,y,z)_AU, (vx,vy,vz)_AU/day)
# The project-relevant measurement (Kepler's 3rd law) is a statistical
# property over many orbits and is insensitive to phase angle.

# ── Dwarf-planet Horizons state vectors, epoch 2026-08-07 00:00 TDB ─────────
# Geometric ecliptic-of-J2000.0 heliocentric states (AU, AU/day), retrieved
# 2026-08-07 from NASA JPL Horizons (DE441). Masses and state vectors come
# directly from this table; the J2000 mean-element table has been removed.
# Format: name -> (mass_kg, (x,y,z)_AU, (vx,vy,vz)_AU/day)
_DWARF_HORIZONS_2026 = {
    "pluto":    (1.303e22,
                 ( 1.911002736447441e+01, -2.870044922610192e+01, -2.582313914053856e+00),
                 (-9.491682518826951e-03, -1.100104682826279e-02, -8.350375769450584e-04)),
    "eris":     (1.66e22,
                 ( 8.440260091082227e+01,  4.037742563223271e+01, -1.729527260481625e+01),
                 (-1.267266445814759e-02, -1.120587853252052e-02,  1.000591411450249e-03)),
    "ceres":    (9.393e20,
                 ( 1.420613483573760e-01,  3.307982340094216e+00, -7.472224330884519e-02),
                 (-2.223094777789978e-02, -9.562308932103390e-03,  1.990894694710314e-03)),
    "makemake": (3.1e21,
                 (-4.664856382616918e+01, -8.789761011780168e+00,  2.407038498854734e+01),
                 (-1.198010761686636e-02, -1.419771820299119e-02, -2.722935247539225e-04)),
    "haumea":   (4.006e21,
                 (-3.742606407117538e+01, -2.326119403775734e+01,  2.351955651455920e+01),
                 (-1.078712461555459e-02, -1.380060721659996e-02, -8.499474934254291e-05)),
}


def _dwarf_planet(name: str) -> dict:
    """Return the dwarf planet's Horizons-2026 heliocentric state as a body dict.

    Uses the geometric ecliptic-of-J2000.0 state vector at epoch
    2026-08-07 00:00 TDB (JD 2461259.5).
    """
    mass_kg, pos_au, vel_au_per_day = _DWARF_HORIZONS_2026[name]
    return {
        "name": name.capitalize(),
        "mass_kg": mass_kg,
        "pos_au": tuple(pos_au),
        "vel_au_per_day": tuple(vel_au_per_day),
    }


# ── Galilean moon data (J2000 heliocentric) ──────────────────────────────────
# Galilean orbital elements (planetocentric) at J2000:
#   a (km), period (days), mass (kg)
# We place the moons at J2000 mean longitudes distributed around their
# orbits so the resulting heliocentric configuration is self-consistent.
_GALILEAN_DATA2 = [
    # name       mass_kg    a_km       period_d   lon_at_j2000_deg
    ("Io",        8.9319e22,    421_800.0,  1.769138,  135.0),
    ("Europa",    4.7998e22,    671_034.0,  3.551181,  225.0),
    ("Ganymede",  1.4819e23,  1_070_412.0,  7.154553,  315.0),
    ("Callisto",  1.0759e23,  1_882_709.0, 16.689018,   45.0),
]


def _galilean_moons_j2000(jupiter: dict) -> list[dict]:
    """
    Build the 4 Galilean moons at their J2000 heliocentric states
    (jupiter's heliocentric state + planetocentric offset).

    The planetocentric orbit is assumed circular and in Jupiter's
    orbital plane; the offset is computed from each moon's `lon_at_j2000`
    (true longitude at J2000). This is a self-consistent initial state
    for the N-body solver; it is *not* the exact J2000 ephemeris
    (which would require a full JPL ephemeris), but it's accurate
    enough for a project-grade validation run.
    """
    out = []
    for name, mass, a_km, _period, lon_deg in _GALILEAN_DATA2:
        a_au = a_km * _KM_TO_AU
        lon  = lon_deg * math.pi / 180.0
        # Position offset from Jupiter in the J2000 ecliptic plane
        # (z component ≈ 0 for a circular orbit in Jupiter's plane).
        dx = a_au * math.cos(lon)
        dy = a_au * math.sin(lon)
        # Circular orbital speed around Jupiter, treating Jupiter's
        # gravity as the dominant local term (μ_J + μ_sun ≈ μ_J since
        # the moons are within ~0.015 AU of Jupiter).
        mu_j = _GM_JUPITER
        v_au_per_day = math.sqrt(mu_j / a_au)
        # Velocity perpendicular to position (prograde).
        vx = -v_au_per_day * math.sin(lon)
        vy =  v_au_per_day * math.cos(lon)
        out.append({
            "name":      f"Jupiter-{name}",   # disambiguate from planet name
            "mass_kg":   mass,
            "pos_au":    (jupiter["pos_au"][0] + dx,
                          jupiter["pos_au"][1] + dy,
                          jupiter["pos_au"][2]),
            "vel_au_per_day": (jupiter["vel_au_per_day"][0] + vx,
                               jupiter["vel_au_per_day"][1] + vy,
                               jupiter["vel_au_per_day"][2]),
        })
    return out


# ── Extended presets ─────────────────────────────────────────────────────────
def _sun_planets_moon_preset() -> dict:
    """
    Sun + 8 planets + Earth's Moon (10 bodies).

    All planet states are the Horizons geometric ecliptic-of-J2000.0
    heliocentric vectors at epoch 2026-08-07 00:00 TDB (JD 2461259.5).
    The Moon's geocentric offset at the same epoch (AU / AU-day) is added
    to Earth's heliocentric state to give the Moon's heliocentric state.
    """
    bodies = [_SUN] + [_planet(n) for n in
                       ["mercury", "venus", "earth", "mars",
                        "jupiter", "saturn", "uranus", "neptune"]]

    # Locate Earth by name rather than by position in the list. This is
    # robust to future reordering or additions (e.g. inserting Pluto
    # between Mars and Jupiter) that would otherwise silently misplace
    # the Moon and give a catastrophic phase error in the OOD test.
    by_name = {b["name"]: b for b in bodies}
    if "Earth" not in by_name:
        raise RuntimeError("sun_planets_moon preset: Earth not in body list")
    bodies.append(_moon_relative_to_earth(by_name["Earth"]))
    return {
        "name": "sun_planets_moon",
        "label": "Sun + 8 planets + Earth's Moon (10 bodies)",
        "bodies": bodies,
        "duration_years": 10,
        "sample_per_year": 12,
        "reference": "leapfrog",
    }


def _full_solar_system_extended_preset() -> dict:
    """
    Sun + 8 planets + Earth's Moon + 5 dwarf planets + 4 Galilean moons
    (19 bodies). The most realistic preset.

    Notes
    -----
    - Planets, the Moon, and the five dwarf planets all use Horizons
      geometric ecliptic-of-J2000.0 heliocentric/geocentric state vectors
      at epoch 2026-08-07 00:00 TDB (JD 2461259.5); no J2000 mean elements
      are propagated.
    - Galilean moons use toy planetocentric circular orbits around
      Jupiter's Horizons-2026 position (real masses and orbital radii),
      because real JUP230 ephemerides are out of scope. See the module
      docstring limitation note.
    """
    bodies = [_SUN] + [_planet(n) for n in
                       ["mercury", "venus", "earth", "mars",
                        "jupiter", "saturn", "uranus", "neptune"]]

    # Locate Earth and Jupiter by name so this preset stays correct if
    # the planet list above is ever reordered or extended. Hard-coded
    # indices (bodies[3] = Earth, bodies[6] = Jupiter) used to silently
    # misplace the Moon and Galilean moons in that case.
    by_name = {b["name"]: b for b in bodies}
    for required in ("Earth", "Jupiter"):
        if required not in by_name:
            raise RuntimeError(
                f"solar_system_extended preset: {required} not in body list"
            )

    # Add Earth's Moon (geocentric offset, J2000).
    bodies.append(_moon_relative_to_earth(by_name["Earth"]))

    # Add dwarf planets (Keplerian heliocentric, J2000).
    for name in ("pluto", "eris", "ceres", "makemake", "haumea"):
        bodies.append(_dwarf_planet(name))

    # Add 4 Galilean moons at J2000 heliocentric (Jupiter + planetocentric).
    for moon_entry in _galilean_moons_j2000(by_name["Jupiter"]):
        bodies.append(moon_entry)

    return {
        "name": "solar_system_extended",
        "label": "Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)",
        "bodies": bodies,
        "duration_years": 10,           # aligned to sun_planets_moon (10 yr / 120 samples) so the
                                       # only variable vs that preset is body count (19 vs 10);
                                       # outer dwarfs won't complete an orbit in this window, their
                                       # Kepler rows are expected to be NaN.
        "sample_per_year": 12,
        "reference": "leapfrog",
    }


# ── In-distribution baseline ─────────────────────────────────────────────────
# The surrogates were trained on 25-body galaxy discs (init_galaxy_disc
# with N=25, masses log-uniform in [0.5, 5] -> mass ratio ~10, and a
# coarse time step DT=0.002). To confirm that OOD generalisation
# failure is due to *distribution shift* and not a pipeline bug, this
# preset reproduces the training distribution as closely as possible:
# same N=25, same mass range [0.5, 5], and the same coarse step
# dt_N = 0.002. In the in-distribution branch dt_N = 1 / sample_per_year
# (the duration_years factor cancels), so sample_per_year=500 gives
# dt_N = 0.002 = training DT; duration_years=5 then fixes a ~5-crossing-
# time horizon (2500 samples). The runner calls `init_galaxy_disc`
# directly for this one; see `ic_loader._load_in_distribution_disc`.
# (An earlier version used a "real IMF" m in [0.1, 50] at dt_N = 0.01,
# which was OOD on both mass ratio (500x vs 10x) and time step (5x vs
# training) -- not a valid in-distribution sanity check.)
_DISC_IMF_PRESET = {
    "name": "disc_imf_in_distribution_baseline",
    "label": "25-body galaxy disc, training IMF (in-distribution sanity)",
    "bodies": None,                    # populated by the loader
    "duration_years": 5,               # crossing-time units (~5 crossings)
    "sample_per_year": 500,            # dt_N = 1/500 = 0.002 = training DT
    "reference": "leapfrog",
    "in_distribution": True,
}


# ── The actual preset list ────────────────────────────────────────────────────
# All helper functions and data tables live above. We construct the
# `PRESETS` list last so every dependency (incl. the extended presets and
# the disc-imf in-distribution baseline) is defined first.
PRESETS: list[dict] = [
    _preset("inner_planets", "Inner planets (Mercury → Mars + Sun)",
            ["mercury", "venus", "earth", "mars"],
            duration_years=10, sample_per_year=12),

    _preset("full_solar_system", "All 8 planets + Sun",
            ["mercury", "venus", "earth", "mars",
             "jupiter", "saturn", "uranus", "neptune"],
            duration_years=200, sample_per_year=12),

    # The Galilean-moon preset is *not* the real Sun-Jupiter system;
    # building it from J2000 moon ephemerides is out of scope. It is
    # replaced with a synthetic-but-realistic Jupiter+4 Galileans entry
    # via `_galilean_moons()` below.
    _preset("jupiter_galileans", "Jupiter + Galilean moons (4-body toy)",
            ["earth"],   # placeholder, overridden below
            duration_years=1, sample_per_year=12),

    _preset("sun_earth_only", "Sun–Earth 2-body (Keplerian reference)",
            ["earth"],
            duration_years=10, sample_per_year=12,
            reference="kepler"),

    # ── Extended Solar System (Sun + 8 planets + Earth's Moon) ─────────────
    # The Moon's J2000 heliocentric state is Earth's state + the Moon's
    # J2000 geocentric state (DE405). This is the cleanest way to
    # include the Moon in a Sun-centred N-body system without dragging
    # in a full ephemeris.
    _sun_planets_moon_preset(),

    # ── Sun + 8 planets + 5 dwarf planets + Moon + 4 Galilean moons ──────
    # The most realistic preset: 19 bodies. Dwarf planets use Keplerian
    # heliocentric elements; the Moon and Galileans are placed at their
    # J2000 heliocentric positions (≈ planet + planetocentric offset).
    # Mass ratios dwarf-Sun ~ 10⁻⁸, so they're tracers, but the
    # gravitational coupling is still there and the system tests the
    # GNN's worst-case multi-scale stress.
    _full_solar_system_extended_preset(),

    _DISC_IMF_PRESET,
]

# Replace the placeholder Galileo preset with the synthetic version
# built from `_galilean_moons()`. The basic `_preset` form is just
# Sun + Earth (since "earth" is a dummy in the placeholder).
PRESETS = [p if p["name"] != "jupiter_galileans" else _galilean_moons()
           for p in PRESETS]


def get_preset(name: str) -> dict:
    """Return the preset with the given slug; raise KeyError if missing."""
    for p in PRESETS:
        if p["name"] == name:
            return p
    raise KeyError(f"unknown preset {name!r}; available: "
                   f"{[p['name'] for p in PRESETS]}")
