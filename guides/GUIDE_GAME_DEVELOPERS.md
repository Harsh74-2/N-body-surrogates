# For Game Developers

You want cheap, believable gravity for a few hundred bodies without a physics
PhD or a per-frame O(N²) loop. Here is what this repo can (and cannot) give
you today.

## The short version

- A learned surrogate can **replace the solver for short stretches** — think
  preview trajectories, background galaxy motion, UI-scale orbits — but it
  **cannot replace it for long-horizon simulation**. The measured rollout
  horizon is the operating limit; respect it and the results look fine,
  exceed it and orbits visibly fall apart.
- The direct O(N²) leapfrog solver in this repo is, honestly, **fast enough
  for game scales on a modern CPU** — it beats every surrogate in
  single-frame inference up to N=100. A learned model only wins once you are
  **batched and on GPU** (the batched MLP beats the solver from N=100 on).
- The genuinely game-relevant regime: **batched prediction of many
  candidate futures** (e.g. plotting 50 possible trajectories of one ship) —
  a single matrix multiply answers what the solver answers one loop at a
  time.

## What ships today

| Asset | Use |
|---|---|
| Trained checkpoints (MLP / LSTM / GNN, N=10–100, stable + single-step variants) | load with PyTorch, query per frame |
| Leapfrog (Störmer–Verlet) reference integrator with Plummer softening | your fallback and your ground truth |
| 6 Solar-System presets with real JPL Horizons initial conditions | drop-in star-system scenes |
| 168 reference-vs-surrogate clips + 336 scrub-able viewers | see exactly where each model breaks |

Browse the [animation gallery](https://harsh74-2.github.io/N-body-surrogates/animations.html)
and the [interactive viewers](https://harsh74-2.github.io/N-body-surrogates/interactive_anim/index.html)
before writing any code — five minutes of scrubbing tells you more than this
page can.

## Practical advice

1. **Pick the GNN for smoothness, the MLP for speed.** The GNN has the lowest
   rollout error slope (orbits stay believable longest); the batched MLP is
   the fastest surrogate. The LSTM's single-step accuracy is great, but its
   N=100 rollout failure is documented — don't ship it blind.
2. **Budget the horizon, not the step.** The models are accurate per step
   (0.3–3.3% of scale length on real solar systems) but compounding
   multiplies that by ~5–6 over a rollout. Decide how many frames you need
   to stay believable, then re-anchor to the solver that often.
3. **Re-anchor often is the real design pattern.** Treat the surrogate as a
   predictor for *in-between* frames and the cheap solver as the authority
   every K frames — this is the "surrogate sits alongside the integrator"
   pattern from the thesis, and it sidesteps the compounding problem
   entirely.
4. **Same weights for any N.** Because the models are per-body and
   shared-weight, a checkpoint trained at N=100 runs at N=37 — you can
   change the population without retraining.
5. **Seeded determinism.** Everything in the pipeline runs with seed 42;
   deterministic replays behave like a game replay buffer would want.

## What would make this a full game toolkit (not built yet)

- A browser sandbox where you upload initial conditions and get the
  surrogate-vs-solver rollout back — the evaluation side already produces all
  the artefacts for it, but the loop is not wired up.
- A CPU/GPU-native ONNX export path with batching; latency measurements here
  are PyTorch, CPU-first, and conservative.
- Training data for game-scale mass distributions (the current models were
  trained on equal-mass discs; your boss fight with 3 planets and 400
  asteroids is a different distribution).

## Honest limits

No collisions, no relativity, no hydrodynamics. Mass ratios of ~10^5 (Sun vs.
satellite moons) are already where the models degrade; two-plus orders beyond
that, use the solver. And always keep the persistence baseline in your
profiler — if your surrogate is not beating "repeat last frame", nothing else
matters.