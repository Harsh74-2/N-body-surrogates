# For ML Educators and Students

A compact, complete case study you can teach from: three neural architectures
learning a physical dynamical system, with every number, checkpoint, and
failure mode published.

## Why this repo works as course material

1. **Every concept you already teach, instantiated.** Overfitting-free
   in-distribution fits, distribution shift, autoregressive error
   compounding, the gap between one-step accuracy and long-horizon
   stability, latency vs. accuracy, weight sharing as a symmetry prior —
   all of it shows up in one project with real numbers.
2. **Small models.** The MLP is 210k parameters, the GNN 168k, the LSTM 866k.
   Students can retrain a cell on a laptop-class GPU in minutes, not days.
3. **The failure modes are visible, not hidden.** The LSTM's N=100 rollout
   failure is documented as a waiver, not buried. The OOD presets where every
   architecture diverges are published as animations.
4. **Metrics done right.** Identity (persistence) baselines next to every MSE,
   explained variance, rollout slope, predictive horizon k* — a worked example
   of "scale-aware" evaluation that most tutorials skip.

## Suggested lesson sequence

1. **Baseline reasoning**: start with the persistence baseline. Why is
   "repeat the last frame" the right null model for a next-state map?
2. **Architecture comparison**: MLP vs LSTM vs GNN on identical data.
   Why does the GNN win the *rollout* metric while the LSTM wins some
   *single-step* cells? (Hint: gravity is a sum of pairwise interactions;
   message passing computes exactly that.)
3. **Compounding demo**: single-step OOD error is 0.3–3.3% of the preset's
   scale length; the rollout mean is 11–14%. Have students compute the
   amplification factor (4.7–6.3) and explain where it comes from.
4. **Regularisation as a physics prior**: contrast single-step vs
   stability-trained checkpoints. The rollout-MSE term lowers the error
   slope in 10 of 12 pairs *without* paying an accuracy tax. Why?
5. **Honest benchmarking**: the latency tables — the direct O(N²) solver
   beats every surrogate in single-frame CPU inference up to N=100. Discuss
   when "ML is faster" is actually true (batched, GPU, N≥100) and when it
   is marketing.

## Point students at

- [Animation gallery](https://harsh74-2.github.io/N-body-surrogates/animations.html)
  — 168 clips: reference solver vs. each surrogate, every N and preset.
- [Interactive viewers](https://harsh74-2.github.io/N-body-surrogates/interactive_anim/index.html)
  — 336 scrub-able viewers for step-by-step inspection.
- `results/all_eval.json` and `results/predictive_horizon.json` for raw
  numbers to re-plot in a notebook.
- `docs/guides/ARCHITECTURE.md` and `docs/guides/TRAINING_GUIDE.md` for the
  technical reference.

## Assignments that work

- Reproduce one cell of the evaluation grid from a released checkpoint and
  report it against the published number.
- Retrain the GNN at N=25 with the stability term switched off and measure
  the change in rollout slope.
- Take one OOD preset where the model diverges and characterise *when* it
  leaves the error envelope — is it a close encounter, an extreme mass
  ratio, or slow drift?

## A note on academic integrity

This project was produced with AI-assisted tooling (disclosed in the thesis).
If you use these materials in teaching, that disclosure itself is a useful
discussion point: what does responsible AI assistance in research look like?