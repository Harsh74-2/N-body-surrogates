# Training Guide

How the checkpoints in this repo were trained, and the practical tips that
came out of getting them there. Read `docs/guides/ARCHITECTURE.md` first for
the model shapes.

## The two training modes

### 1. Single-step (the baseline variant)

Plain supervised learning on the next-state map:

- **Loss**: mean squared error between predicted and leapfrog-computed next
  state, per coordinate.
- **Optimiser**: Adam with early stopping on validation loss.
- **Windows**: sliding window of W=5 past frames, stride 1, per body.
- **Determinism**: every run uses seed 42, so reported cells are one
  deterministic run, not an ensemble.

### 2. Stability-trained variant

Same architecture, one added term. The loss becomes

```
L = L_single_step + w_rollout * L_rollout        (w_rollout = 0.1)
```

where `L_rollout` is the MSE of a K=5-step autoregressive rollout. No
explicit energy loss term is used (w_energy = 0) — the rollout term alone
regularises the dynamics.

Two schedule details that matter:

- **Warm-up, then ramp**: the rollout weight starts at 0 and is ramped up
  after a warm-up period of pure single-step training. Naive warm-starting
  with the rollout term active wrecks the early fit.
- **Checkpoint selection by total validation loss** (single-step + weighted
  rollout), not by single-step validation MSE alone. Selecting on single-step
  MSE undoes most of the stability benefit.

## Results this produced

- Stable variants **match or beat their siblings on single-step MSE** in most
  cells — the rollout term acts as a structural regulariser, not an accuracy
  tax.
- They **lower the rollout error slope in 10 of 12** (architecture, N) pairs.
- On the OOD survivor mean they stay within a few points of the sibling in
  10 of 11 comparable cells.

## Hints, tips, and tricks (the part you'd otherwise learn the hard way)

1. **Batch size and memory (GNN at N=100).** The GNN's rollout path was the
   memory peak. Two things fixed OOMs: batch size 96 (not 128/256) at
   N=100, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2. **Gradient clipping `max_norm=1.0`** on all trainers — the rollout path
   occasionally produces large gradients through the autoregressive branch;
   clipping keeps Adam's state sane.
3. **Identity baseline discipline.** Always report the persistence
   (repeat-last-frame) baseline next to your MSE. MSE scales with N and with
   the data's absolute scale; the identity baseline is what makes a 2e-4
   meaningful or meaningless.
4. **Warm-up clamp.** Warm-up is clamped to W=5 frames; longer warm-ups are
   rejected by the MLP/LSTM trainers because a longer window than the model
   was trained on is a silent shape error.
5. **Evaluate the rollout, not just the step.** A checkpoint can look
   excellent per-step and diverge in tens of frames. The repo's stability
   harness (rollout slope, predictive horizon k*, persistence-crossover step)
   exists because single-step MSE alone mis-ranks checkpoints.
6. **Ensembles and calibration knobs did not help.** Longer warm-up, a
   two-model ensemble, and per-body linear-drift calibration were all tried
   at inference time; none lowered the best single-model mean error. The
   single best checkpoint per cell is what ships.

## Reproducing a cell

```bash
# single-step variant
python gnn_train.py --preset disc --N 100

# stability-trained variant (same command family via the launcher)
bash train_stable_variants.sh
```

Full runbook: `RETRAIN_RUNBOOK.md` in the repo root.

## Known open failure

The LSTM at N=100 develops rollout error faster than every other cell
(predictive horizon k* = 39/45 steps for the two variants). It is documented
as a gate waiver in the thesis, not fixed. If you retrain it, that cell is the
one to watch.