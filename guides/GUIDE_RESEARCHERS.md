# For Researchers and Engineers

What a practitioner in simulation, HPC, or applied ML should take from this
repo — the measured results, the evaluation harness, and the reusable
tooling.

## The measured headline

Three per-body, shared-weight surrogates (MLP 210k, LSTM 866k, GNN 168k
parameters) trained on synthetic galaxy discs, evaluated against a symplectic
leapfrog solver, in-distribution and on real Solar-System initial conditions
(NASA JPL Horizons, epoch 2026-08-07):

- **Single-step**: accurate everywhere in this study. In-distribution MSE in
  the 1e-8 band for the right (model, N) cells; OOD single-step error
  0.3–3.3% of each preset's scale length.
- **Compounding is the dominant cost**: autoregressive feedback multiplies
  per-step error by 4.7–6.3; 4 of 6 OOD presets leave the half-scale error
  envelope within the 128-step horizon; the no-compounding OOD mean is
  11.3–13.7% for all three architectures — architecture choice barely moves
  the OOD ceiling.
- **Architecture ranking is metric-dependent**: the GNN is the best rollout
  citizen (lowest slope of all 24 cells, never crosses the persistence
  baseline in-distribution); the LSTM wins some single-step cells but fails
  late-horizon at N=100 (documented waiver, k* = 39/45 steps).
- **Stability training works**: a rollout-MSE term (weight 0.1, no energy
  term, warm-up-then-ramp, total-validation-loss selection) lowers the
  rollout error slope in 10 of 12 pairs with no single-step accuracy tax.
- **Latency is a regime, not a crossover**: the direct O(N²) solver wins
  single-frame CPU inference up to N=100; the batched MLP wins from N=100.

## Why this matters beyond N-body

The transferable finding is methodological: **per-step accuracy does not
predict autoregressive fidelity, and the identity baseline is the only honest
yardstick for either.** Any project that iterates a learned one-step model
(states, weather, traffic, molecular dynamics) inherits this structure: the
same compounding, the same need for a rollout metric, the same failure of
single-step model selection.

## The reusable evaluation harness

- **Rollout slope + predictive horizon k\***: the first rollout step where
  the model's MSE overtakes the persistence baseline — a single number that
  ranks checkpoints by how long they stay useful, not by how well they fit.
- **Identity-baseline discipline** in every table.
- **Cross-N audit framework**: the same preset grid re-scored at every
  training count, so architecture claims are checked for N-dependence.
- **OOD presets with two error lanes**: a no-compounding lane (windows
  rebuilt from the reference each frame) and an autoregressive lane — this
  separation is what isolates compounding from per-step transfer failure.
- **Gate scripts**: training gates (per-checkpoint) that reject divergence
  or spatial collapse, with documented waivers rather than silent passes.

## Tooling you can lift

| Tool | What it does |
|---|---|
| `real_case_validation/presets.py` | preset samplers, disc + 6 solar-system presets (Horizons ICs) |
| leapfrog reference integrator | symplectic, validated, Plummer-softened |
| `train_stable_variants.sh` + `RETRAIN_RUNBOOK.md` | the full stability-variant recipe |
| `results/cross_N_audit.md` + per-N audit mds | the audit-report format itself |
| site build (`pages` branch) | 168 clips + 336 scrub-able viewers from result dumps |

## Suggested follow-ups

- GPU-batched latency sweep (the CPU measurements are conservative for
  surrogates by construction).
- Probabilistic heads (calibrated uncertainty per frame) — the natural
  answer to "when can I trust the rollout?"
- Hierarchical message passing for mixed-scale systems
  (`solar_system_extended` is the standing stress test).
- A user-submitted-ICs web sandbox: the evaluation pipeline already produces
  every artefact needed; what's missing is the service loop.

## Reproducibility statement

All headline cells are single deterministic runs (seed 42); the latency
benchmark reports the mean of 40 repeats. Numbers in the thesis are generated
as LaTeX macros from the result JSONs, so prose and tables cannot drift. The
full OOD grid, per-N audits, and gate logs ship in `results/`.