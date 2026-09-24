# Architecture Reference

How the three surrogates in this repo are built, and why they share one design.

## The shared design: per-body, shared weights

Every model is a **next-state map**: it reads a short history of the simulation
and predicts every body's position and velocity one integration step
(`dt = 0.01`) into the future, in 3D.

The key design choice is that **one set of weights is applied to each body
in turn**. The network never sees N as a quantity — it sees one body at a
time, plus whatever per-body context its architecture can gather. Consequences:

- The parameter count is **independent of N**. The same checkpoint trained at
  N=10 runs at N=100 without resizing.
- Weight sharing acts as a symmetry prior: gravity treats bodies equally, and
  so does the model.

| Model | Structure | Parameters |
|---|---|---|
| MLP | width 256, depth 4, flattened W=5 window | 210,182 |
| LSTM | hidden 256, 2 stacked layers, per-body sequence over W=5 frames | 865,542 |
| GNN | hidden 128, 2 message-passing steps (per-body shared-weight node update + GRU-style aggregation) | 168,199 |

## What each model is good at (measured, not vibes)

See `results/` and the [live site](https://harsh74-2.github.io/N-body-surrogates/)
for every number; the short version at a glance:

- **GNN** — the best *rollout* citizen. Lowest rollout error slope of all 24
  (architecture, variant, N) cells and never crosses the persistence baseline
  within the 128-step in-distribution horizon. Smallest parameter count.
- **LSTM** — best *single-step* accuracy in the tightest cells (reaches the
  1e-8 MSE band at N=25 and N=100), and the only consistent beneficiary of
  stability training at small N — but one documented late-horizon compounding
  failure at N=100.
- **MLP** — the latency winner once inference is batched (beats the direct
  solver from N=100 on), and the simplest baseline.

## Data contract

- **Inputs**: sliding window of W=5 frames, stride 1; per-body features are
  position and velocity (3D each), with mass available as context.
- **Output**: per-body position + velocity at the next frame.
- **Training data**: synthetic galaxy discs (roughly equal-mass bodies on
  Plummer-profile initial conditions) produced by a validated symplectic
  leapfrog (Störmer–Verlet) integrator.
- **Evaluation data**: the same discs held out, plus six real Solar-System
  presets whose initial conditions come from the NASA JPL Horizons ephemeris
  (epoch 2026-08-07) — this is the out-of-distribution test.

## Where to look in the code

| File | Role |
|---|---|
| `mlp_train.py`, `lstm_train.py`, `gnn_train.py` | one trainer per architecture |
| `real_case_validation/presets.py` | preset definitions (disc + 6 solar-system presets) |
| `train_stable_variants.sh` | the stability-variant training launcher |
| `results/all_eval.json`, `results/predictive_horizon.json` | headline metric dumps |

Numbers in the thesis prose are generated as LaTeX macros from these dumps —
see `metrics.tex` in the Overleaf bundle — so tables and prose cannot drift
apart.

## Honest limits

- The surrogates approximate the next-state map; they are **not** replacements
  for a symplectic integrator where long-term energy conservation matters.
- Autoregressive rollout compounds error: the per-step OOD error is 0.3–3.3%
  of the preset's scale length, but the rollout mean error across presets sits
  at 11–14% once compounding is allowed, and 4 of 6 OOD presets leave the
  error envelope within the horizon.
- Trained on discs, not Keplerian orbits: transfer to the Solar System is a
  distribution-shift measurement, not a demonstration of physics understanding.