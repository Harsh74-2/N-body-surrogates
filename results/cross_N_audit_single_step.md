# Cross-N audit (single-step)



Post-retrain (2026-09-22).

Single-step test metrics (held-out windows, in-distribution):
mean squared error of the next-frame prediction, the
energy-relative drift of that prediction, and the per-step
latency. Source: `results/all_eval.json`.

| N | variant | params | single-step MSE | energy drift | latency (ms) |
|---|---|---|---|---|---|
| 10 | MLP | 210,182 | 7.79e-08 | 7.11e-04 | 0.256 |
| 10 | MLP (stable) | 210,182 | 8.28e-08 | 6.09e-04 | 0.253 |
| 10 | LSTM | 865,542 | 1.47e-07 | 9.93e-04 | 0.387 |
| 10 | LSTM (stable) | 865,542 | 1.51e-07 | 1.07e-03 | 0.383 |
| 10 | GNN | 168,199 | 2.71e-06 | 2.13e-03 | 2.118 |
| 10 | GNN (stable) | 168,199 | 2.55e-06 | 2.18e-03 | 2.016 |
| 25 | MLP | 210,182 | 7.75e-08 | 1.74e-04 | 0.251 |
| 25 | MLP (stable) | 210,182 | 8.29e-08 | 1.93e-04 | 0.254 |
| 25 | LSTM | 865,542 | 1.20e-08 | 8.36e-05 | 0.384 |
| 25 | LSTM (stable) | 865,542 | 1.16e-08 | 9.48e-05 | 0.454 |
| 25 | GNN | 168,199 | 1.51e-06 | 1.06e-03 | 2.152 |
| 25 | GNN (stable) | 168,199 | 1.15e-06 | 8.16e-04 | 3.889 |
| 50 | MLP | 210,182 | 1.20e-07 | 9.51e-05 | 0.250 |
| 50 | MLP (stable) | 210,182 | 1.30e-07 | 1.07e-04 | 0.248 |
| 50 | LSTM | 865,542 | 2.26e-06 | 6.90e-04 | 0.451 |
| 50 | LSTM (stable) | 865,542 | 1.84e-06 | 6.54e-04 | 0.788 |
| 50 | GNN | 168,199 | 8.38e-08 | 2.91e-04 | 7.326 |
| 50 | GNN (stable) | 168,199 | 8.05e-08 | 2.68e-04 | 16.315 |
| 100 | MLP | 210,182 | 2.15e-08 | 5.42e-05 | 0.251 |
| 100 | MLP (stable) | 210,182 | 1.91e-08 | 4.90e-05 | 0.315 |
| 100 | LSTM | 865,542 | 5.30e-08 | 8.48e-05 | 0.790 |
| 100 | LSTM (stable) | 865,542 | 4.77e-08 | 8.94e-05 | 1.514 |
| 100 | GNN | 168,199 | 2.83e-07 | 2.62e-04 | 33.077 |
| 100 | GNN (stable) | 168,199 | 4.04e-08 | 8.30e-05 | 67.660 |

Per-preset single-step mean error (% of the preset's scale

length L), averaged over every start index in the window:

each prediction is scored against a reference-rebuilt input

window, so errors do not compound. Source: the fresh

2026-09-22 single-step dumps.

### disc_imf_in_distribution_baseline (in-distribution)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 0.00% | 0.00% | 0.00% | 0.00% |
| MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| LSTM | 0.00% | 0.00% | 0.02% | 0.00% |
| LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| GNN | 0.02% | 0.03% | 0.01% | 0.03% |
| GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; |

### full_solar_system (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 0.78% | 1.00% | 1.05% | 1.06% |
| MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| LSTM | 2.95% | 3.34% | 1.39% | 0.98% |
| LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| GNN | 0.89% | 1.08% | 1.02% | 1.07% |
| GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; |

### inner_planets (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 29.37% | 29.51% | 29.63% | 29.44% |
| MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| LSTM | 29.13% | 29.40% | 29.75% | 29.56% |
| LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| GNN | 29.71% | 29.85% | 29.87% | 29.81% |
| GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; |

### jupiter_galileans (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 11.05% | 11.10% | 11.25% | 11.27% |
| MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| LSTM | 10.67% | 10.76% | 11.33% | 11.13% |
| LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| GNN | 11.27% | 11.39% | 11.44% | 11.35% |
| GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; |

### solar_system_extended (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 0.54% | 1.97% | 1.35% | 1.13% |
| MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| LSTM | 11.37% | 9.70% | 1.45% | 1.20% |
| LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| GNN | 0.31% | 0.63% | 0.44% | 0.69% |
| GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; |

### sun_earth_only (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 25.26% | 25.31% | 25.45% | 25.49% |
| MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| LSTM | 24.76% | 24.67% | 25.54% | 25.44% |
| LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| GNN | 25.52% | 25.63% | 25.84% | 25.59% |
| GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; |

### sun_planets_moon (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 0.83% | 1.07% | 1.14% | 1.19% |
| MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| LSTM | 3.30% | 3.82% | 1.57% | 1.09% |
| LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; |
| GNN | 0.97% | 1.20% | 1.10% | 1.19% |
| GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; |

