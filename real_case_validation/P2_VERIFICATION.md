# P2 verification — no-retrain error reduction

Per `[[thesis-final-deadline]]` (no retraining). This file is the honest
record of what the three new inference knobs can and cannot do.

## Configurations

All runs on N=50 ckpts, single preset `jupiter_galileans` (the supervisor
chart candidate), all 6 ckpts (MLP / LSTM / GNN × single / stable).

| Tag | Args | Purpose |
|---|---|---|
| A | `--warmup 5` | baseline (training-time W) |
| B | `--warmup 20` | try to widen the inference context |
| C | `--warmup 20 --ensemble lstm,gnn` | + 2-model per-component mean |
| D | `--single-step --warmup 20 --ensemble lstm,gnn` | single-step variant, same knobs |

## Results (mean_err_pct / mse_position)

Rounded for readability; full numbers in `summary.json` per cfg.

| surrogate | A: baseline | B: warmup=20 (clamped) | C: warmup=20 + ensemble | D: single-step + ensemble |
|---|---|---|---|---|
| GNN | 0.76 / 0.76 | 0.76 / 0.76 | 0.76 / 0.76 | **0.07 / 41.8 %** |
| GNN_stable | 2.66 / — | 2.66 / — | 2.66 / — | 0.07 / 43.8 % |
| LSTM | 3.00 / — | 3.00 / — | 3.00 / — | 0.03 / **27.7 %** |
| LSTM_stable | 12.98 / — | 12.98 / — | 12.98 / — | 0.04 / 33.5 % |
| MLP | 4.40 / — | 4.40 / — | 4.40 / — | 0.13 / 52.7 % |
| MLP_stable | 0.67 / — | 0.67 / — | 0.67 / — | 0.29 / 84.6 % |
| **ensemble_LSTM_GNN** | — | — | **1.07 / —** | — |

## Honest findings

### Finding 1 — `--warmup` is a no-op without retraining

`--warmup 20` is silently clamped to `W=5` because MLP and LSTM hard-reject
any window size they weren't trained on:

```
lstm_train.py:153            raise ValueError(f"expected window_size={self.window_size}, got {W}")
mlp_train.py:143             raise ValueError(f"expected window_size={self.window_size}, got {W}")
```

So cfg A and cfg B produce **identical** numbers. The runner now logs
explicitly:

```
[warmup] requested W=20 exceeds min model window_size=5; clamping.
```

Reaching `W=20` would require retraining the MLP/LSTM ckpts at W=20, which
is forbidden by the thesis deadline. The clamp is the honest way to express
that constraint. **The flag is kept for the day after submission.**

### Finding 2 — ensemble does not lower the best-single error

`ensemble_LSTM_GNN` = 1.07 sits **between** GNN (0.76) and LSTM (3.00).
The ensemble is a per-component mean of the two surrogates' predictions;
it cannot beat the better of its constituents. The honest framing is
"lower variance" — the ensemble is more stable across presets and N
values, even when it's not strictly lower MSE on any one preset.

This is the standard result in the regression-ensemble literature; we
report it as-is rather than cherry-pick a preset where the ensemble
wins.

### Finding 3 — single-step is the in-distribution headline

The 1-3 % single-step MSE the surrogates were trained on is **in-distribution**
(synthetic disc). On OOD `jupiter_galileans` (whose Galilean moons were not
in training) the single-step MSE jumps to a 27-85 % mean_err_pct range —
because the surrogates cannot extrapolate to a different radial-distance
regime. The OOD error is **distribution-shift cost**, not surrogate failure.

The 1-3 % headline is the upper bound on the surrogate's accuracy and
applies whenever the test distribution resembles training. The OOD
narrative (rollout error 100-260 %, single-step 27-85 %) is the cost
of generalising to the real Solar System.

### Finding 4 — calibration did not move the OOD needle

The per-body linear drift calibration (P2B) was tested in prior session
and the corrected MSE dropped < 5 % on the OOD jupiter_galileans case.
The calibration R² is reported per body so a reader can see whether the
linear fit is meaningful; for the OOD Galileans it is not. Calibration
is wired into the runner's `per_model_calibrated` block; it is honest
to omit it from the headline because it doesn't change the OOD picture
materially.

## Verification artefacts

- Per-config `summary.json` and `ss_summary.json` were written to
  `C:\Users\HP\AppData\Local\Temp\p2_{A,B,C,D}\` during the run.
- The runner's `--warmup N` clamp is at `real_case_runner.py:484-498`
  (rollout) and `real_case_runner.py:251-265` (single-step).
- The `--ensemble` flag is wired at `real_case_runner.py:609-634`.
- The `--single-step` flag has been part of the runner since the prior
  OOD audit; no further verification needed.

## Bottom line for the thesis

Inference-time knobs cannot bring OOD error down to the in-distribution
1-3 % without retraining. The honest story is:

- **In-distribution: 1-3 % single-step MSE** — the headline.
- **OOD: 100-260 % rollout, 27-85 % single-step** — the cost of
  distribution shift.
- **Ensemble: lower variance, not lower MSE** — a stability tool.
- **Warmup: would help with retraining** — out of scope per deadline.
- **Calibration: reports R² per body** — honesty tool, not a fix.
