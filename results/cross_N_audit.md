# Cross-N audit (rollout)



Post-audit retrain (2026-09). Every number below is read

directly from the canonical result JSONs of that run; nothing is

hand-transcribed.

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
| 10 | MLP | 144.7% | 38.0% | 62.2% | 57.0% | 3.2% | 62.5% | 13.5% | 39.4% |
| 10 | MLP (stable) | 144.7% | 38.1% | 62.2% | 57.0% | 3.2% | 62.5% | 13.5% | 39.4% |
| 10 | LSTM | 145.2% | 60.3% | 63.0% | 67.3% | 3.2% | 63.6% | 14.0% | 45.2% |
| 10 | LSTM (stable) | 145.2% | 59.6% | 63.0% | 66.9% | 3.2% | 63.5% | 14.0% | 45.0% |
| 10 | GNN | 138.3% | 68.3% | 62.8% | 70.3% | div. | 63.8% | 14.2% | 55.9% |
| 10 | GNN (stable) | 138.3% | 68.6% | 62.8% | 70.3% | 3.4% | 63.8% | 14.2% | 47.2% |
| 25 | MLP | 148.6% | 38.1% | 62.4% | 57.7% | 3.2% | 62.6% | 13.5% | 39.6% |
| 25 | MLP (stable) | 149.1% | 37.9% | 62.5% | 57.7% | 3.2% | 62.6% | 13.5% | 39.5% |
| 25 | LSTM | 144.3% | 40.1% | 62.1% | 57.2% | 3.4% | 62.5% | 13.5% | 39.8% |
| 25 | LSTM (stable) | 144.6% | 39.7% | 62.1% | 57.0% | 3.4% | 62.5% | 13.5% | 39.7% |
| 25 | GNN | 148.1% | 32.2% | 62.1% | 55.1% | 3.3% | 62.2% | 13.4% | 38.1% |
| 25 | GNN (stable) | 147.8% | 32.5% | 62.1% | 55.2% | 3.3% | 62.3% | 13.4% | 38.1% |
| 50 | MLP | 152.5% | 37.2% | 62.3% | 55.9% | 3.7% | 62.4% | 13.4% | 39.1% |
| 50 | MLP (stable) | 152.5% | 37.0% | 62.3% | 55.9% | 3.7% | 62.4% | 13.4% | 39.1% |
| 50 | LSTM | 150.8% | 41.2% | 62.1% | 56.4% | 3.8% | 62.4% | 13.5% | 39.9% |
| 50 | LSTM (stable) | 151.0% | 41.4% | 62.1% | 56.5% | 3.9% | 62.4% | 13.5% | 40.0% |
| 50 | GNN | 151.3% | 39.4% | 62.2% | 56.1% | 3.8% | 62.4% | 13.5% | 39.6% |
| 50 | GNN (stable) | 151.3% | 39.4% | 62.2% | 56.1% | 3.8% | 62.4% | 13.5% | 39.6% |
| 100 | MLP | 150.0% | 35.6% | 62.4% | 56.7% | 3.3% | 62.4% | 13.5% | 39.0% |
| 100 | MLP (stable) | 150.3% | 36.2% | 62.4% | 57.0% | 3.2% | 62.5% | 13.5% | 39.1% |
| 100 | LSTM | 150.2% | 33.9% | 62.3% | 56.0% | 3.3% | 62.4% | 13.4% | 38.6% |
| 100 | LSTM (stable) | 149.9% | 33.9% | 62.3% | 56.1% | 3.2% | 62.4% | 13.4% | 38.6% |
| 100 | GNN | 150.7% | 35.9% | 62.4% | 56.8% | 3.3% | 62.5% | 13.5% | 39.0% |
| 100 | GNN (stable) | 149.5% | 34.8% | 62.4% | 56.4% | 3.2% | 62.4% | 13.5% | 38.8% |

### In-distribution rollout drift (test split)

Mean energy-relative rollout drift over a K=50 autoregressive
rollout on held-out test windows (dimensionless), from the
post-audit evaluation of every checkpoint.

| N | variant | rollout drift (K=50) |
|---|---|---|
| 10 | MLP | 2.11e-05 |
| 10 | MLP (stable) | 7.41e-06 |
| 10 | LSTM | 1.42e-05 |
| 10 | LSTM (stable) | 6.22e-06 |
| 10 | GNN | 1.76e-05 |
| 10 | GNN (stable) | 1.59e-06 |
| 25 | MLP | 7.11e-06 |
| 25 | MLP (stable) | 1.85e-06 |
| 25 | LSTM | 4.84e-06 |
| 25 | LSTM (stable) | 1.09e-06 |
| 25 | GNN | 2.64e-06 |
| 25 | GNN (stable) | 1.05e-05 |
| 50 | MLP | 1.85e-06 |
| 50 | MLP (stable) | 8.13e-07 |
| 50 | LSTM | 5.32e-07 |
| 50 | LSTM (stable) | 5.32e-07 |
| 50 | GNN | 1.16e-06 |
| 50 | GNN (stable) | 1.10e-06 |
| 100 | MLP | 4.62e-06 |
| 100 | MLP (stable) | 3.73e-07 |
| 100 | LSTM | 2.96e-07 |
| 100 | LSTM (stable) | 1.46e-07 |
| 100 | GNN | 6.40e-07 |
| 100 | GNN (stable) | 4.40e-06 |
