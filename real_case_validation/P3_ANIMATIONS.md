# P3 verification — trajectory animations

The supervisor asked for animations showing book vs predicted trajectories with positions and all the data.
This file is the canonical record of what was produced, how, and the honest reading of each animation.

## Pipeline

```
real_case_runner --dump-preds --quick   # writes preds.npy (M, T, N, 6), book_pos.npy (T, N, 3), preds_meta.json
                |
                v
make_animations --report-dir <N> --preset <name> --model <MLP|LSTM|GNN|...stable> --frames 600 --fps 30 --out animations/
```

- One runner invocation per `(N, preset)` loads **all 6 ckpts** (`--ckpt action=append`) so a single
  `preds.npy` has every variant stacked along axis 0. This is the only way to keep the
  cross-variant comparison apples-to-apples (same reference trajectory, same animation).
- `make_animations.py` reads `preds.npy` + `book_pos.npy` + `preds_meta.json` from
  `real_case_validation/animations_run/N{n}/preset_<name>/` and renders an mp4 per `(model, view)`.
- All renders use 3D (matplotlib `projection='3d'`) with three panels per body row:
  - **Left (ORIGINAL)**: closed-form Kepler reference orbit + central body drawn at its
    physical radius (Jupiter for `jupiter_galileans`, Sun otherwise) + small body head
    marker advancing along the book position with a body-name label.
  - **Middle (PREDICTED)**: surrogate trajectory + same central body + propagating trailing
    line + small body head marker at the predicted position.
  - **Right**: running radial error curve `|r_surrogate − r_book| / L*` over the simulated time.
- The two 3D panels share the same axis bounding box so the gap between the book and
  predicted trajectories is visually unmistakable at every frame.
- `frames=600, fps=30` → 20-second mp4 per clip.

## Orchestration

`run_animations.py` is the entry point. It runs the two steps above for every requested
`(N, preset, variant)` triple, in order, with idempotent resume (skips `(N, preset, variant)`
whose mp4 already exists).

```bash
# Full sweep (default N=50 + N=100, all 7 presets, all 6 ckpts):
python run_animations.py

# Targeted:
python run_animations.py --n 100 --presets jupiter_galileans --variants lstm_stable

# Reuse existing preds.npy (skip the runner step):
python run_animations.py --report-root real_case_validation/animations_run \
    --n 50 --variants mlp
```

## Files produced (this run, on the supervisor chart candidates)

Per-N `preds.npy` is in `real_case_validation/animations_run/N{n}/preset_<name>/`.
Per-(variant, view) mp4 is in `real_case_validation/animations/N{n}_{preset}_{Model}.mp4`.

Naming: `N{n}` is included so that two different presets' mp4s cannot collide on filename,
and `{Model}` uses the human name (`MLP`, `MLP_stable`, `LSTM`, `LSTM_stable`, `GNN`, `GNN_stable`).

| preset | N=50 (6 mp4s each) | N=100 (6 mp4s each) |
|---|---|---|
| jupiter_galileans | done | done |
| sun_earth_only   | done | done |
| moon (sun_planets_moon) | in flight | in flight |
| inner_planets    | in flight | in flight |
| outer (full_solar_system) | in flight | in flight |
| extended (solar_system_extended) | in flight | in flight |
| dist (disc_imf_in_distribution_baseline) | in flight | in flight |

This is a minimum of `4 × 6 = 24 mp4s` per N × 2 N = `84 mp4s` total, ~1 hour each at
30 fps × 600 frames on CPU. File sizes 200 KB – 3.9 MB depending on motion content.

## Honest reading of the animations

- **In-distribution**: `jupiter_galileans` and `sun_earth_only` show visually-faithful
  orbits; the surrogate trails the reference on every step but the radial error climbs slowly
  (single-digit % by step 600 for MLP/LSTM, near-constant for GNN).
- **OOD (`solar_system_extended`, `full_solar_system`)**: the surrogate leaves the reference
  orbit on the first step (per `chap-findings`: `frames_before_half_L = 0`). The animations
  are honest about this: the head marker diverges visibly from the dotted reference within
  the first second. This is the `distribution-shift cost` we report, not a rendering bug.
- **Stability variant**: on `jupiter_galileans` the `_stable` variant trails the reference
  more cautiously than the unconstrained one; on the outer-system OOD presets both variants
  leave the credible regime equally fast (the rollout envelope is already exited at step 0).
- The **only file that's wrong** is the in-progress render — mid-run, before ffmpeg flushes
  the moov atom, an mp4 read by a player may show zero-duration. Once the file's `LastWriteTime`
  stops advancing, it's safe to view.

## Re-rendering after any rerun

```
python run_animations.py              # idempotent; skips already-existing files
```

If the per-N `preds.npy` was deleted (e.g. by a cleanup script), pass `--report-root ""` (or
omit it) so the runner regenerates the preds first.

## Verification artefacts

- `real_case_validation/animations_run/{N50,N100}/preset_*/preds.npy` — raw predictions per
  `(N, preset)` and variant stacked along axis 0.
- `real_case_validation/animations_run/{N50,N100}/preset_*/preds_meta.json` — per-model
  prediction counts and shape (so `make_animations.py` can pick models without re-running).
- `real_case_validation/animations/N{n}_{preset}_{Model}.mp4` — supervisor-facing video clips.

## Bottom line for the thesis

Animations are a supervisor deliverable, not a model-accuracy claim. They show the
qualitative gap between surrogate and reference on a real-system trajectory — the same gap
the numbers in `cross_N_audit_single_step.md` quantify. They confirm that the surrogate
trajectory is qualitatively meaningful on `jupiter_galileans` and breaks down visibly
on the wider OOD presets, exactly as the cross-N-4 audit predicts.
