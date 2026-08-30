# Cross-N Real-Case Validation, Error-Percentage Audit

Autoregressive rollout variant: each surrogate predicts **forward in time** from its own previous output. Errors compound over the rollout. Use this report to read the distribution-shift cost of the Solar System relative to the synthetic disc training set.

Each cell is the *mean error %* averaged across every preset that ran for that N. The cell on the right (N=100) is the **best** any model in this family can do on the given training budget; the cell on the left (N=10) is the worst. Reading left-to-right should show the gradual improvement as the training budget grows.

## Headline: mean error % by (N, model)

| model | N=10 | N=25 | N=50 | N=100 | Δ (N=100 − N=10, pp) |
|---|---|---|---|---|---|
| MLP | 261.1 % | 296.5 % | 237.7 % | 195.4 % | -65.7 |
| MLP_stable | 238.5 % | 152.2 % | 129.2 % | 115.9 % | -122.6 |
| LSTM | 236.2 % | 165.1 % | 213.9 % | 158.7 % | -77.5 |
| LSTM_stable | 186.4 % | 198.9 % | 329.0 % | 178.0 % | -8.3 |
| GNN | 109.3 % | 168.0 % | 134.7 % | 209.7 % | +100.4 |
| GNN_stable | 154.7 % | 119.3 % | 238.2 % | 219.3 % | +64.6 |

## Per-preset mean error % across N (headline preset detail)

### `disc_imf_in_distribution_baseline` (in-distribution)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 286.35 % | 428.72 % | 365.56 % | 227.79 % |
| MLP_stable | 286.83 % | 235.29 % | 186.69 % | 183.14 % |
| LSTM | 316.80 % | 250.70 % | 219.81 % | 222.30 % |
| LSTM_stable | 201.17 % | 242.50 % | 295.24 % | 232.16 % |
| GNN | 176.16 % | 353.44 % | 176.26 % | 295.93 % |
| GNN_stable | 200.99 % | 154.44 % | 300.17 % | 234.19 % |

### `full_solar_system` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 410.41 % | 581.63 % | 284.79 % | 267.36 % |
| MLP_stable | 422.54 % | 185.88 % | 117.15 % | 99.27 % |
| LSTM | 395.47 % | 241.58 % | 355.43 % | 166.71 % |
| LSTM_stable | 255.62 % | 294.95 % | 757.95 % | 288.19 % |
| GNN | 125.90 % | 248.82 % | 197.14 % | 366.39 % |
| GNN_stable | 208.55 % | 125.34 % | 344.46 % | 293.06 % |

### `inner_planets` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 121.21 % | 78.67 % | 94.33 % | 109.64 % |
| MLP_stable | 104.37 % | 105.21 % | 119.39 % | 108.84 % |
| LSTM | 116.50 % | 88.53 % | 88.18 % | 94.41 % |
| LSTM_stable | 124.08 % | 100.13 % | 91.93 % | 109.41 % |
| GNN | 76.72 % | 77.16 % | 73.12 % | 87.23 % |
| GNN_stable | 106.95 % | 77.83 % | 94.45 % | 78.30 % |

### `jupiter_galileans` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 384.08 % | 405.87 % | 328.99 % | 280.67 % |
| MLP_stable | 394.48 % | 188.46 % | 129.99 % | 111.88 % |
| LSTM | 422.65 % | 189.85 % | 264.65 % | 147.75 % |
| LSTM_stable | 188.16 % | 248.22 % | 492.48 % | 166.69 % |
| GNN | 156.27 % | 194.53 % | 123.46 % | 298.78 % |
| GNN_stable | 199.62 % | 123.36 % | 253.76 % | 191.16 % |

### `solar_system_extended` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 267.24 % | 311.89 % | 262.86 % | 218.48 % |
| MLP_stable | 166.53 % | 142.63 % | 115.77 % | 90.35 % |
| LSTM | 125.21 % | 178.94 % | 257.70 % | 222.79 % |
| LSTM_stable | 193.11 % | 212.38 % | 323.87 % | 181.65 % |
| GNN | 89.59 % | 132.22 % | 168.14 % | 199.30 % |
| GNN_stable | 138.24 % | 174.98 % | 350.83 % | 422.95 % |

### `sun_earth_only` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 142.59 % | 94.19 % | 135.79 % | 127.72 % |
| MLP_stable | 146.81 % | 129.08 % | 136.79 % | 133.22 % |
| LSTM | 154.59 % | 125.65 % | 118.47 % | 132.63 % |
| LSTM_stable | 193.08 % | 142.28 % | 131.91 % | 144.69 % |
| GNN | 81.13 % | 84.52 % | 87.21 % | 112.65 % |
| GNN_stable | 147.49 % | 99.85 % | 124.14 % | 94.76 % |

### `sun_planets_moon` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 215.74 % | 174.52 % | 191.78 % | 135.89 % |
| MLP_stable | 147.88 % | 79.21 % | 98.72 % | 84.40 % |
| LSTM | 122.29 % | 80.11 % | 193.39 % | 124.11 % |
| LSTM_stable | 149.29 % | 151.67 % | 209.85 % | 123.45 % |
| GNN | 59.37 % | 85.15 % | 117.60 % | 107.81 % |
| GNN_stable | 80.94 % | 79.64 % | 199.47 % | 220.81 % |

## Family-level verdict (single vs stable, mean across N)

| family | N=10 single | N=10 stable | Δ (pp) | N=100 single | N=100 stable | Δ (pp) |
|---|---|---|---|---|---|---|
| MLP | 261.09 % | 238.49 % | -22.60 | 195.36 % | 115.87 % | -79.49 |
| LSTM | 236.22 % | 186.36 % | -49.86 | 158.67 % | 178.03 % | +19.36 |
| GNN | 109.30 % | 154.68 % | +45.38 | 209.73 % | 219.32 % | +9.59 |
