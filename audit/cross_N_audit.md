# Cross-N Real-Case Validation, Error-Percentage Audit

Autoregressive rollout variant: each surrogate predicts **forward in time** from its own previous output. Errors compound over the rollout. Use this report to read the distribution-shift cost of the Solar System relative to the synthetic disc training set.

Each cell is the *mean error %* averaged across every preset that ran for that N. The cell on the right (N=100) is the **best** any model in this family can do on the given training budget; the cell on the left (N=10) is the worst. Reading left-to-right should show the gradual improvement as the training budget grows.

## Headline: mean error % by (N, model)

| model | N=10 | N=25 | N=50 | N=100 | Δ (N=100 − N=10, pp) |
|---|---|---|---|---|---|
| MLP | 262.8 % | 297.1 % | 238.3 % | 196.4 % | -66.4 |
| MLP_stable | 239.5 % | 153.0 % | 130.0 % | 116.6 % | -122.8 |
| LSTM | 233.7 % | 165.9 % | 216.4 % | 158.6 % | -75.1 |
| LSTM_stable | 187.3 % | 198.4 % | 329.9 % | 179.3 % | -8.0 |
| GNN | 108.4 % | 168.6 % | 135.1 % | 210.7 % | +102.3 |
| GNN_stable | 154.5 % | 120.5 % | 238.7 % | 219.5 % | +65.0 |

## Per-preset mean error % across N (headline preset detail)

### `disc_imf_in_distribution_baseline` (in-distribution)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 286.35 % | 428.72 % | 365.56 % | 227.79 % |
| MLP_stable | 286.83 % | 235.29 % | 186.69 % | 183.14 % |
| LSTM | 316.80 % | 250.70 % | 219.81 % | 222.30 % |
| LSTM_stable | 201.17 % | 242.50 % | 295.24 % | 232.16 % |
| GNN | 176.42 % | 353.44 % | 176.26 % | 295.93 % |
| GNN_stable | 201.17 % | 154.44 % | 300.17 % | 234.19 % |

### `full_solar_system` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| MLP | 410.40 % | 581.37 % | 284.78 % | 267.44 % |
| MLP_stable | 422.84 % | 185.88 % | 117.15 % | 99.26 % |
| LSTM | 394.82 % | 241.58 % | 361.90 % | 158.26 % |
| LSTM_stable | 255.60 % | 283.03 % | 758.55 % | 288.20 % |
| GNN | 114.17 % | 248.74 % | 197.21 % | 368.38 % |
| GNN_stable | 204.57 % | 125.46 % | 344.49 % | 290.79 % |

### `inner_planets` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|
| MLP | 120.70 % | 78.51 % | 94.49 % | 110.06 % |
| MLP_stable | 104.15 % | 104.67 % | 119.35 % | 108.82 % |
| LSTM | 116.61 % | 88.22 % | 87.80 % | 94.04 % |
| LSTM_stable | 123.22 % | 100.25 % | 91.63 % | 109.02 % |
| GNN | 76.67 % | 77.59 % | 73.18 % | 86.77 % |
| GNN_stable | 107.37 % | 77.84 % | 94.51 % | 78.78 % |

### `jupiter_galileans` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|
| MLP | 384.15 % | 405.86 % | 328.97 % | 280.52 % |
| MLP_stable | 394.58 % | 188.46 % | 129.99 % | 111.88 % |
| LSTM | 398.96 % | 189.83 % | 264.65 % | 147.75 % |
| LSTM_stable | 188.17 % | 248.22 % | 492.48 % | 166.70 % |
| GNN | 156.24 % | 194.54 % | 123.49 % | 298.79 % |
| GNN_stable | 199.54 % | 123.36 % | 253.76 % | 190.39 % |

### `solar_system_extended` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|
| MLP | 279.62 % | 316.76 % | 267.44 % | 224.78 % |
| MLP_stable | 172.60 % | 148.37 % | 121.12 % | 95.88 % |
| LSTM | 131.92 % | 185.36 % | 264.40 % | 230.69 % |
| LSTM_stable | 200.61 % | 218.60 % | 329.10 % | 191.12 % |
| GNN | 96.09 % | 135.98 % | 170.84 % | 204.41 % |
| GNN_stable | 140.71 % | 182.32 % | 353.90 % | 427.92 % |

### `sun_earth_only` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|
| MLP | 142.59 % | 94.19 % | 135.79 % | 127.72 % |
| MLP_stable | 146.81 % | 129.08 % | 136.79 % | 133.22 % |
| LSTM | 154.59 % | 125.65 % | 118.47 % | 132.63 % |
| LSTM_stable | 193.08 % | 142.28 % | 131.91 % | 144.69 % |
| GNN | 81.13 % | 84.52 % | 87.21 % | 112.65 % |
| GNN_stable | 147.49 % | 99.85 % | 124.14 % | 94.76 % |

### `sun_planets_moon` (OOD)

| model | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|
| MLP | 215.76 % | 174.60 % | 191.42 % | 136.60 % |
| MLP_stable | 148.66 % | 79.11 % | 98.74 % | 84.35 % |
| LSTM | 122.35 % | 79.66 % | 197.52 % | 124.45 % |
| LSTM_stable | 149.20 % | 153.81 % | 210.60 % | 123.27 % |
| GNN | 57.89 % | 85.05 % | 117.51 % | 107.97 % |
| GNN_stable | 80.73 % | 79.97 % | 199.60 % | 219.73 % |

## Family-level verdict (single vs stable, mean across N)

| family | N=10 single | N=10 stable | Δ (pp) | N=100 single | N=100 stable | Δ (pp) |
|---|---|---|---|---|---|---|
| MLP | 262.79 % | 239.49 % | -23.30 | 196.42 % | 116.65 % | -79.77 |
| LSTM | 233.72 % | 187.29 % | -46.43 | 158.59 % | 179.31 % | +20.72 |
| GNN | 108.37 % | 154.51 % | +46.14 | 210.70 % | 219.51 % | +8.81 |