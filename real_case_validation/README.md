# Real-Case Validation Pipeline

This package evaluates the trained MLP / LSTM / GNN N-body surrogates
**on real Solar-System initial conditions**, producing a separate
self-contained validation report at `report/`.

The original pipeline (training + project benchmark) is **untouched**.
The surrogates are loaded from the existing `training_runs/*/model_best.pt`
checkpoints and run unmodified on rescaled real-world ICs.

## Quick start

From the repo root:

```bash
python -m real_case_validation.real_case_runner \
    --ckpt training_runs/gnn_20260628-171838/model_best.pt:gnn \
    --ckpt training_runs/lstm_20260628-171335/model_best.pt:lstm \
    --ckpt training_runs/mlp_20260628-213252/model_best.pt:mlp
```

This runs every built-in preset and writes:

```
real_case_validation/report/
├── real_case_report.md           # human-readable summary
├── real_case_report.json         # machine-readable
├── dashboard.png                 # cross-preset trajectory + energy
└── preset_<name>/
    ├── trajectory.png            # xy projection: reference vs surrogates
    ├── energy.png                # energy drift per integrator
    └── summary.json              # per-preset metrics
```

To run a single preset, add `--preset sun_earth_only`. To add `--quick`
for a faster smoke test, see below.

## What "validation against real life" actually means

The surrogates were trained on 25-body galaxy discs
(`simulation_3d.init_galaxy_disc`, mass ratios ≲ 10). The Solar System
is a **very different distribution**: 8–10 bodies with mass ratios
of 10⁵ (Sun:Earth) or higher. So the numbers in `real_case_report.md`
measure **out-of-distribution generalisation**, not domain fit.

The runner also ships an in-distribution baseline preset
(`disc_imf_in_distribution_baseline`) built directly in the trained
dimensionless units from the same `init_galaxy_disc` generator and
training mass range (mass ratios ≲ 10, m ∈ [0.5, 5] in code units).
Its errors should sit at the in-distribution level seen in the
`report_N*/single_step/single_step_report.md` tables; if they don't,
the warm-up or the checkpoint loading is broken and the OOD numbers
are not meaningful.

## Built-in presets

| name | what | duration | reference |
|---|---|---|---|
| `inner_planets` | Sun + Mercury → Mars | 10 yr | leapfrog |
| `full_solar_system` | Sun + 8 planets | 200 yr | leapfrog |
| `jupiter_galileans` | Jupiter + Io/Europa/Ganymede/Callisto (toy circular orbits) | 1 yr | leapfrog |
| `sun_earth_only` | Sun + Earth (Keplerian) | 10 yr | kepler (analytical) |
| `sun_planets_moon` | Sun + 4 rocky planets + Jupiter/Saturn + Moon | 27 yr | leapfrog |
| `solar_system_extended` | Sun + 8 planets + Moon + Galileans (19 bodies) | 27 yr | leapfrog |
| `disc_imf_in_distribution_baseline` | synthetic 25-body disc (in-distribution) | 5 crossing times | leapfrog |

The planetary initial states are NASA JPL Horizons geometric
ecliptic-of-J2000.0 state vectors at epoch 2026-08-07 (JD 2461259.5),
retrieved from the Horizons system; dwarf-planet states and the
Galilean-moon toy orbits are documented in `presets.py`.

## Adding a custom IC

Drop a JSON file anywhere and pass `--preset path/to/your_ic.json`. The
schema is exactly the per-preset dict from `presets.py`:

```json
{
  "name":  "my_binary",
  "label": "Custom binary star system",
  "bodies": [
    {"name": "M1", "mass_kg": 1.5e30,
     "pos_au": [0, 0, 0], "vel_au_per_day": [0, 0, 0]},
    {"name": "M2", "mass_kg": 0.8e30,
     "pos_au": [5.0, 0, 0], "vel_au_per_day": [0, 1.0, 0]}
  ],
  "duration_years": 100,
  "sample_per_year": 12
}
```

The runner validates the schema, rescales the IC into N-body units
(M = Σm so Σm=1, L = a characteristic radius with per-preset overrides),
runs the reference + surrogates, and writes the per-preset artefacts into
`<out>/preset_<name>/` (default `--out real_case_validation/report`;
the committed reports use `report_N{n}`).

To add a new *built-in* preset, append a dict to `PRESETS` in
`presets.py` and re-run the runner.

## How the rescaling works

`unit_rescale.scale_for_preset` picks:

- **M = total system mass** in kg, so Σ mass_N = 1 exactly
  (matches the training convention).
- **L = outermost body semi-major axis** in metres (heliocentric for
  planet presets, Jupiter's orbit for the Galilean preset, the disc
  scale radius for the in-distribution baseline).
- **T = √(L³ / (G·M))** derived.

The runner then converts:

- pos_N = pos_SI / L
- vel_N = vel_SI / √(G·M/L)
- mass_N = mass_SI / M
- dt_N = dt_SI / T

…and reports the rescaling factors in each preset's `summary.json`
under `scale: {M_kg, L_m, T_s, …}` so the conversion is auditable.

The system's COM velocity is subtracted before rescaling, so the
network sees a zero-momentum state (matching the training distribution
of discs in centrifugal equilibrium).

## Smoke test

```bash
python -m real_case_validation.real_case_runner \
    --ckpt training_runs/gnn_20260628-171838/model_best.pt:gnn \
    --preset sun_earth_only --quick
```

This runs the smallest 2-body preset with reduced reference
sub-stepping, finishes in <30 s on CPU, and writes
`report/preset_sun_earth_only/{trajectory,energy}.png` +
`summary.json`.

## Files

| file | purpose |
|---|---|
| `presets.py` | Built-in real-Solar-System scenarios + Galilean toy + disc baseline |
| `ic_loader.py` | Validates and rescales preset / custom JSON ICs |
| `unit_rescale.py` | SI → dimensionless-unit conversion (auditable) |
| `references.py` | High-precision leapfrog + closed-form Kepler reference |
| `metrics.py` | MSE, energy drift, normalised trajectory error |
| `plots.py` | Per-preset + cross-preset dashboard plots |
| `real_case_runner.py` | CLI entry point |

## Out of scope (deliberate)

- **No retraining**. The existing checkpoints are used as-is. Fine-tuning
  the surrogates on Solar-System data is a separate work item.
- **No runtime ephemeris dependency** (jplephem, REBOUND, network calls).
  The initial states were retrieved from NASA Horizons once and are
  hardcoded in `presets.py`; the reference trajectory is our own
  high-precision leapfrog, keeping the pipeline self-contained and
  offline.
- **No GPU requirements**. CPU works for every preset (≤25 bodies ×
  ~500 frames per preset).
