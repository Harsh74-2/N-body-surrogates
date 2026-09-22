# Cross-N audit (rollout)



Post-retrain (2026-09-22). Every number below is read

directly from the canonical result JSONs of that run; nothing is

hand-transcribed.

### Rollout OOD mean (headline)

Mean rollout positional error (% of scale length L) averaged
over the six out-of-distribution presets. '>100' = every OOD
preset diverged for that cell, so no survivor mean exists.

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 66.6% | 68.9% | 64.6% | 65.6% |
| MLP (stable) | 65.0% | 68.7% | 64.6% | 66.6% |
| LSTM | 64.7% | 70.8% | 64.0% | &gt;100 |
| LSTM (stable) | 63.6% | 74.3% | 65.2% | 97.9% |
| GNN | 69.3% | 66.7% | 73.0% | 72.2% |
| GNN (stable) | 69.1% | 66.6% | 73.1% | 82.5% |

### Out-of-distribution rollout error

Mean rollout positional error as a percentage of the preset's
scale length L, averaged over the rollout, for every training
body count N and model variant. All presets except the disc
baseline are out-of-distribution; their values use the calibrated
post-hoc error of the thesis. Cells marked div. diverged
(error above 100% of L) and are excluded from the mean; the disc
baseline is in-distribution and also excluded from the mean.

| N | variant | disc (in-dist.) | full solar system | inner planets | jupiter galileans | solar system extended | sun earth only | sun planets moon | OOD mean |
|---|---|---|---|---|---|---|---|---|---|
| 10 | MLP | 171.9% | div. | 64.6% | div. | div. | 68.5% | div. | 66.6% |
| 10 | MLP (stable) | 167.5% | div. | 63.1% | div. | div. | 67.0% | div. | 65.0% |
| 10 | LSTM | 204.9% | div. | 64.6% | div. | div. | 64.8% | div. | 64.7% |
| 10 | LSTM (stable) | 327.2% | div. | 63.2% | div. | div. | 64.0% | div. | 63.6% |
| 10 | GNN | 122.3% | div. | 64.6% | div. | div. | 74.1% | div. | 69.3% |
| 10 | GNN (stable) | 120.6% | div. | 64.1% | div. | div. | 74.0% | div. | 69.1% |
| 25 | MLP | 134.1% | div. | 68.1% | div. | div. | 69.8% | div. | 68.9% |
| 25 | MLP (stable) | 123.7% | div. | 67.7% | div. | div. | 69.7% | div. | 68.7% |
| 25 | LSTM | 3314.3% | div. | 70.5% | div. | div. | 71.1% | div. | 70.8% |
| 25 | LSTM (stable) | 730.6% | div. | 74.3% | div. | div. | 74.4% | div. | 74.3% |
| 25 | GNN | 167.5% | div. | 63.0% | div. | div. | 70.3% | div. | 66.7% |
| 25 | GNN (stable) | 147.1% | div. | 62.6% | div. | div. | 70.6% | div. | 66.6% |
| 50 | MLP | 166.4% | div. | 64.2% | div. | div. | 65.1% | div. | 64.6% |
| 50 | MLP (stable) | 146.7% | div. | 64.1% | div. | div. | 65.1% | div. | 64.6% |
| 50 | LSTM | 130.2% | div. | 63.1% | div. | div. | 65.0% | div. | 64.0% |
| 50 | LSTM (stable) | 126.6% | div. | 64.2% | div. | div. | 66.1% | div. | 65.2% |
| 50 | GNN | 176.9% | div. | 66.6% | div. | div. | 79.4% | div. | 73.0% |
| 50 | GNN (stable) | 161.8% | div. | 66.4% | div. | div. | 79.8% | div. | 73.1% |
| 100 | MLP | 287.8% | div. | 63.0% | div. | div. | 68.3% | div. | 65.6% |
| 100 | MLP (stable) | 185.3% | div. | 63.7% | div. | div. | 69.4% | div. | 66.6% |
| 100 | LSTM | 315.3% | div. | div. | div. | div. | div. | div. | &mdash; |
| 100 | LSTM (stable) | 300.8% | div. | 97.9% | div. | div. | div. | div. | 97.9% |
| 100 | GNN | 114.8% | div. | 64.2% | div. | div. | 80.1% | div. | 72.2% |
| 100 | GNN (stable) | 179.5% | div. | 73.0% | div. | div. | 92.0% | div. | 82.5% |

### Error with vs. without autoregressive feedback (OOD mean)

The same checkpoints scored two ways, averaged over the six
out-of-distribution presets. 'Single-step (no compounding)'
rebuilds the input window from the true reference at every step,
so prediction errors never feed back into the model -- it is
scored at every point of the rollout horizon without feedback.
'Full rollout' lets predictions feed back (autoregressive), so
errors compound. Diverged rollout cells (error above 100% of L)
are excluded from the rollout mean, so the compounding factor is
a lower bound. The gap between the two columns isolates the cost
of error compounding: the same checkpoints hold roughly 5x lower
error when their own predictions never feed back.

| N | variant | no compounding | full rollout | factor |
|---|---|---|---|---|
| 10 | MLP | 11.3% | 66.6% | &times;5.9 |
| 10 | MLP (stable) | &mdash; | 65.0% | &mdash; |
| 10 | LSTM | 13.7% | 64.7% | &times;4.7 |
| 10 | LSTM (stable) | &mdash; | 63.6% | &mdash; |
| 10 | GNN | 11.4% | 69.3% | &times;6.1 |
| 10 | GNN (stable) | &mdash; | 69.1% | &mdash; |
| 25 | MLP | 11.7% | 68.9% | &times;5.9 |
| 25 | MLP (stable) | &mdash; | 68.7% | &mdash; |
| 25 | LSTM | 13.6% | 70.8% | &times;5.2 |
| 25 | LSTM (stable) | &mdash; | 74.3% | &mdash; |
| 25 | GNN | 11.6% | 66.7% | &times;5.7 |
| 25 | GNN (stable) | &mdash; | 66.6% | &mdash; |
| 50 | MLP | 11.6% | 64.6% | &times;5.6 |
| 50 | MLP (stable) | &mdash; | 64.6% | &mdash; |
| 50 | LSTM | 11.8% | 64.0% | &times;5.4 |
| 50 | LSTM (stable) | &mdash; | 65.2% | &mdash; |
| 50 | GNN | 11.6% | 73.0% | &times;6.3 |
| 50 | GNN (stable) | &mdash; | 73.1% | &mdash; |
| 100 | MLP | 11.6% | 65.6% | &times;5.7 |
| 100 | MLP (stable) | &mdash; | 66.6% | &mdash; |
| 100 | LSTM | 11.6% | &mdash; | &mdash; |
| 100 | LSTM (stable) | &mdash; | 97.9% | &mdash; |
| 100 | GNN | 11.6% | 72.2% | &times;6.2 |
| 100 | GNN (stable) | &mdash; | 82.5% | &mdash; |

### In-distribution rollout drift (test split)

Mean energy-relative rollout drift over a K=50 autoregressive
rollout on held-out test windows (dimensionless), from the
post-audit evaluation of every checkpoint.

| N | variant | rollout drift (K=50) |
|---|---|---|
| 10 | MLP | 9.31e-02 |
| 10 | MLP (stable) | 7.45e-02 |
| 10 | LSTM | 1.27e-01 |
| 10 | LSTM (stable) | 2.08e-01 |
| 10 | GNN | 1.14e-02 |
| 10 | GNN (stable) | 3.40e-02 |
| 25 | MLP | 1.87e-02 |
| 25 | MLP (stable) | 1.68e-02 |
| 25 | LSTM | 8.74e-03 |
| 25 | LSTM (stable) | 3.54e-02 |
| 25 | GNN | 3.86e-03 |
| 25 | GNN (stable) | 7.59e-03 |
| 50 | MLP | 1.35e-02 |
| 50 | MLP (stable) | 1.16e-02 |
| 50 | LSTM | 1.78e-02 |
| 50 | LSTM (stable) | 2.08e-02 |
| 50 | GNN | 6.26e-03 |
| 50 | GNN (stable) | 3.86e-03 |
| 100 | MLP | 6.31e-03 |
| 100 | MLP (stable) | 9.91e-03 |
| 100 | LSTM | 5.66e-03 |
| 100 | LSTM (stable) | 3.91e-03 |
| 100 | GNN | 1.12e-03 |
| 100 | GNN (stable) | 8.91e-04 |
