# Cross-N Real-Case Validation, Error-Percentage Audit

Single-step variant: each surrogate predicts the next frame only from a warm-up window of leapfrog frames. Errors do **not** compound because the window is always re-built from the reference. This is the headline 1-3 % single-step MSE the surrogates were trained on.

Each cell is the *mean error %* averaged across every preset that ran for that N. The cell on the right (N=100) is the **best** any model in this family can do on the given training budget; the cell on the left (N=10) is the worst. Reading left-to-right should show the gradual improvement as the training budget grows.

## Headline: mean error % by (N, model)

| model | N=10 | N=25 | N=50 | N=100 | Δ (N=100 − N=10, pp) |
|---|---|---|---|---|---|
| MLP | 55.1 % | 79.9 % | 73.2 % | 67.0 % | +11.8 |
| MLP_stable | 72.5 % | 122.7 % | 117.7 % | 84.6 % | +12.1 |
| LSTM | 37.8 % | 42.2 % | 66.8 % | 56.0 % | +18.2 |
| LSTM_stable | 68.2 % | 50.6 % | 79.3 % | 73.7 % | +5.5 |
| GNN | 70.1 % | 64.4 % | 67.2 % | 66.7 % | -3.4 |
| GNN_stable | 61.3 % | 70.3 % | 84.7 % | 118.1 % | +56.8 |

## Per-preset mean error % across N (headline preset detail)

### `disc_imf_in_distribution_baseline` (in-distribution)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 3.66 % | 2.54 % | 1.95 % | 2.73 % |
| MLP_stable | 5.73 % | 2.69 % | 2.81 % | 3.20 % |
| LSTM | 3.73 % | 1.59 % | 1.62 % | 1.87 % |
| LSTM_stable | 8.64 % | 2.57 % | 3.10 % | 2.75 % |
| GNN | 3.77 % | 1.43 % | 1.59 % | 1.94 % |
| GNN_stable | 3.38 % | 1.40 % | 2.42 % | 2.21 % |

### `full_solar_system` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 49.00 % | 60.40 % | 69.53 % | 63.88 % |
| MLP_stable | 47.40 % | 107.90 % | 110.76 % | 73.94 % |
| LSTM | 22.63 % | 29.89 % | 79.20 % | 58.15 % |
| LSTM_stable | 73.00 % | 48.80 % | 95.42 % | 90.60 % |
| GNN | 69.75 % | 41.44 % | 60.97 % | 51.72 % |
| GNN_stable | 39.65 % | 52.78 % | 81.95 % | 136.46 % |

### `inner_planets` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 65.19 % | 73.11 % | 71.39 % | 77.72 % |
| MLP_stable | 94.27 % | 113.39 % | 120.19 % | 112.24 % |
| LSTM | 58.10 % | 53.67 % | 51.85 % | 51.15 % |
| LSTM_stable | 67.60 % | 57.69 % | 62.82 % | 60.42 % |
| GNN | 76.17 % | 77.75 % | 77.47 % | 80.85 % |
| GNN_stable | 78.90 % | 77.60 % | 78.56 % | 77.82 % |

### `jupiter_galileans` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 41.08 % | 50.04 % | 52.66 % | 54.29 % |
| MLP_stable | 61.10 % | 79.26 % | 84.65 % | 86.90 % |
| LSTM | 32.10 % | 28.05 % | 27.73 % | 28.66 % |
| LSTM_stable | 37.93 % | 30.09 % | 33.48 % | 33.66 % |
| GNN | 40.99 % | 42.27 % | 41.84 % | 45.38 % |
| GNN_stable | 43.36 % | 41.82 % | 43.78 % | 42.33 % |

### `solar_system_extended` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 103.85 % | 232.73 % | 168.03 % | 119.44 % |
| MLP_stable | 135.48 % | 302.15 % | 252.03 % | 111.70 % |
| LSTM | 54.06 % | 85.12 % | 155.22 % | 129.38 % |
| LSTM_stable | 122.94 % | 91.86 % | 170.00 % | 150.74 % |
| GNN | 126.24 % | 144.95 % | 122.53 % | 123.03 % |
| GNN_stable | 119.91 % | 162.22 % | 192.43 % | 315.58 % |

### `sun_earth_only` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 67.82 % | 74.39 % | 75.10 % | 80.15 % |
| MLP_stable | 110.94 % | 126.27 % | 128.96 % | 122.31 % |
| LSTM | 68.89 % | 63.64 % | 61.65 % | 57.06 % |
| LSTM_stable | 87.80 % | 70.96 % | 82.92 % | 77.08 % |
| GNN | 96.95 % | 98.46 % | 98.19 % | 107.90 % |
| GNN_stable | 100.75 % | 99.19 % | 103.04 % | 101.08 % |

### `sun_planets_moon` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 55.26 % | 66.02 % | 74.05 % | 70.48 % |
| MLP_stable | 52.45 % | 126.96 % | 124.74 % | 81.56 % |
| LSTM | 24.88 % | 33.36 % | 90.26 % | 65.44 % |
| LSTM_stable | 79.65 % | 52.22 % | 107.63 % | 100.53 % |
| GNN | 76.55 % | 44.61 % | 67.81 % | 55.99 % |
| GNN_stable | 43.27 % | 56.83 % | 90.42 % | 151.18 % |

## Family-level verdict (single vs stable, mean across N)

| family | N=10 single | N=10 stable | Δ (pp) | N=100 single | N=100 stable | Δ (pp) |
|---|---|---|---|---|---|---|
| MLP | 55.12 % | 72.48 % | +17.36 | 66.96 % | 84.55 % | +17.59 |
| LSTM | 37.77 % | 68.22 % | +30.45 | 55.96 % | 73.68 % | +17.72 |
| GNN | 70.06 % | 61.32 % | -8.74 | 66.69 % | 118.09 % | +51.41 |
