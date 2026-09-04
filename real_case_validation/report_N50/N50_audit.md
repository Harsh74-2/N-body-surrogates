# Real-Case Validation, N = 50

Standalone audit for the **N = 50** training-budget rerun. Two complementary modes are reported:

- **Autoregressive rollout** (in `preset_*/summary.json`) — each surrogate predicts forward from its own previous output. Errors compound; the per-step prediction becomes the *warm-up window* for the next. This is the **stress test** for the transfer question: *how far can the model extrapolate before it loses the orbit?*
- **Single-step variant** (in `preset_*/ss_summary.json`) — each surrogate predicts the next frame *only*, with the warm-up window always re-built from the leapfrog reference (never from the model's own output). Errors do not compound. This is the **bare prediction error** and the headline 1-3 % number the surrogates were trained on.

## Headline (autoregressive rollout, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran, normalised by L.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 365.6 % | 217.1 % | 238.3 % |
| MLP_stable | 186.7 % | 120.5 % | 130.0 % |
| LSTM | 219.8 % | 215.8 % | 216.4 % |
| LSTM_stable | 295.2 % | 335.7 % | 329.9 % |
| GNN | 176.3 % | 128.2 % | 135.1 % |
| GNN_stable | 300.2 % | 228.4 % | 238.7 % |

## Headline (single-step, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran. The in-distribution row should sit at 1-3 % — this is the **bare** prediction error the surrogates were trained on.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 1.9 % | 85.1 % | 73.2 % |
| MLP_stable | 2.8 % | 136.9 % | 117.7 % |
| LSTM | 1.6 % | 77.7 % | 66.8 % |
| LSTM_stable | 3.1 % | 92.0 % | 79.3 % |
| GNN | 1.6 % | 78.1 % | 67.2 % |
| GNN_stable | 2.4 % | 98.4 % | 84.7 % |

## Stable vs single, per family (rollout)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 217.15 % | 120.52 % | -96.62 |
| LSTM | 215.79 % | 335.71 % | +119.92 |
| GNN | 128.24 % | 228.40 % | +100.16 |

## Stable vs single, per family (single-step)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 85.13 % | 136.89 % | +51.76 |
| LSTM | 77.65 % | 92.04 % | +14.39 |
| GNN | 78.13 % | 98.36 % | +20.23 |

## Per-preset detail (autoregressive rollout)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 365.56 % | 774.72 % | 48 | 2.36e+00 | 5.376e+00 |
| MLP_stable | 186.69 % | 557.74 % | 30 | 2.10e+00 | 1.449e+00 |
| LSTM | 219.81 % | 691.66 % | 66 | 2.65e+00 | 2.382e+00 |
| LSTM_stable | 295.24 % | 739.95 % | 37 | 2.01e+00 | 3.634e+00 |
| GNN | 176.26 % | 613.81 % | 90 | 2.52e+00 | 1.417e+00 |
| GNN_stable | 300.17 % | 680.67 % | 70 | 1.74e+01 | 3.712e+00 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 284.78 % | 520.11 % | 0 | 1.61e+02 | 3.006e+00 |
| MLP_stable | 117.15 % | 504.95 % | 0 | 3.69e+00 | 5.482e-01 |
| LSTM | 361.90 % | 528.98 % | 0 | 3.90e+01 | 4.909e+00 |
| LSTM_stable | 758.55 % | 1166.79 % | 0 | 4.64e+00 | 2.389e+01 |
| GNN | 197.21 % | 426.51 % | 0 | 1.15e+02 | 1.803e+00 |
| GNN_stable | 344.49 % | 959.30 % | 0 | 8.09e+02 | 5.186e+00 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 94.49 % | 221.83 % | 0 | 6.94e+01 | 4.261e-01 |
| MLP_stable | 119.35 % | 298.21 % | 0 | 3.91e+00 | 5.685e-01 |
| LSTM | 87.80 % | 234.21 % | 0 | 7.60e+01 | 3.536e-01 |
| LSTM_stable | 91.63 % | 233.61 % | 0 | 2.29e+00 | 3.839e-01 |
| GNN | 73.18 % | 225.36 % | 0 | 7.36e+01 | 2.714e-01 |
| GNN_stable | 94.51 % | 237.99 % | 0 | 2.36e+02 | 3.910e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 328.97 % | 734.57 % | 0 | 6.46e+01 | 4.397e+00 |
| MLP_stable | 129.99 % | 259.07 % | 0 | 3.86e+00 | 6.693e-01 |
| LSTM | 264.65 % | 519.36 % | 2 | 7.03e+01 | 3.004e+00 |
| LSTM_stable | 492.48 % | 1022.84 % | 2 | 2.24e+00 | 1.298e+01 |
| GNN | 123.49 % | 478.04 % | 0 | 7.10e+01 | 7.588e-01 |
| GNN_stable | 253.76 % | 646.50 % | 0 | 2.24e+02 | 2.662e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 267.44 % | 1057.55 % | 0 | 4.33e+01 | 2.909e+00 |
| MLP_stable | 121.12 % | 498.83 % | 0 | 3.83e+00 | 6.364e-01 |
| LSTM | 264.40 % | 555.05 % | 0 | 3.97e+01 | 2.803e+00 |
| LSTM_stable | 329.10 % | 645.73 % | 0 | 2.33e+00 | 4.069e+00 |
| GNN | 170.84 % | 447.02 % | 0 | 4.73e+01 | 1.315e+00 |
| GNN_stable | 353.90 % | 1014.41 % | 0 | 1.17e+02 | 6.229e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 135.79 % | 205.50 % | 0 | 6.94e+01 | 7.374e-01 |
| MLP_stable | 136.79 % | 258.20 % | 0 | 3.92e+00 | 7.275e-01 |
| LSTM | 118.47 % | 220.36 % | 0 | 7.61e+01 | 5.514e-01 |
| LSTM_stable | 131.91 % | 196.09 % | 0 | 2.29e+00 | 6.622e-01 |
| GNN | 87.21 % | 223.32 % | 0 | 5.64e+01 | 3.787e-01 |
| GNN_stable | 124.14 % | 219.04 % | 0 | 2.36e+02 | 6.073e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 191.42 % | 516.35 % | 0 | 5.53e+02 | 1.639e+00 |
| MLP_stable | 98.74 % | 504.95 % | 0 | 3.91e+00 | 4.391e-01 |
| LSTM | 197.52 % | 471.46 % | 0 | 3.80e+01 | 2.012e+00 |
| LSTM_stable | 210.60 % | 499.43 % | 0 | 4.57e+00 | 2.115e+00 |
| GNN | 117.51 % | 426.58 % | 0 | 9.70e+01 | 8.894e-01 |
| GNN_stable | 199.60 % | 958.98 % | 0 | 1.37e+03 | 2.920e+00 |

## Per-preset detail (single-step)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 1.95 % | 12.68 % | 8.05e-02 | 1.925e-04 |
| MLP_stable | 2.81 % | 17.24 % | 1.13e-01 | 3.576e-04 |
| LSTM | 1.62 % | 11.81 % | 3.07e-02 | 1.295e-04 |
| LSTM_stable | 3.10 % | 24.41 % | 1.75e-01 | 4.568e-04 |
| GNN | 1.59 % | 9.08 % | 5.00e-02 | 1.195e-04 |
| GNN_stable | 2.42 % | 13.42 % | 5.21e-02 | 2.595e-04 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 69.53 % | 460.04 % | 8.44e-01 | 4.044e-01 |
| MLP_stable | 110.76 % | 513.38 % | 5.03e+00 | 9.630e-01 |
| LSTM | 79.20 % | 390.53 % | 3.04e-01 | 5.357e-01 |
| LSTM_stable | 95.42 % | 340.18 % | 1.89e+00 | 6.877e-01 |
| GNN | 60.97 % | 349.67 % | 1.34e+00 | 3.564e-01 |
| GNN_stable | 81.95 % | 550.40 % | 1.69e+00 | 7.618e-01 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 71.39 % | 132.89 % | 3.05e+02 | 2.253e-01 |
| MLP_stable | 120.19 % | 215.36 % | 2.79e+03 | 6.037e-01 |
| LSTM | 51.85 % | 81.85 % | 2.23e+02 | 1.114e-01 |
| LSTM_stable | 62.82 % | 103.25 % | 1.15e+03 | 1.700e-01 |
| GNN | 77.47 % | 138.81 % | 8.80e+02 | 2.795e-01 |
| GNN_stable | 78.56 % | 140.94 % | 7.73e+02 | 2.805e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 52.66 % | 128.77 % | 6.21e+00 | 1.336e-01 |
| MLP_stable | 84.65 % | 169.12 % | 6.13e+01 | 2.911e-01 |
| LSTM | 27.73 % | 45.25 % | 4.54e+00 | 2.895e-02 |
| LSTM_stable | 33.48 % | 61.61 % | 2.48e+01 | 4.065e-02 |
| GNN | 41.84 % | 65.33 % | 1.93e+01 | 7.127e-02 |
| GNN_stable | 43.78 % | 65.68 % | 1.63e+01 | 7.415e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 168.03 % | 467.14 % | 1.18e+00 | 1.343e+00 |
| MLP_stable | 252.03 % | 517.33 % | 2.40e+00 | 2.863e+00 |
| LSTM | 155.22 % | 456.68 % | 1.17e+00 | 1.190e+00 |
| LSTM_stable | 170.00 % | 423.11 % | 1.63e+00 | 1.363e+00 |
| GNN | 122.53 % | 401.82 % | 1.57e+00 | 8.010e-01 |
| GNN_stable | 192.43 % | 629.98 % | 1.62e+00 | 2.080e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 75.10 % | 150.10 % | 1.08e+03 | 3.187e-01 |
| MLP_stable | 128.96 % | 262.03 % | 9.86e+03 | 9.150e-01 |
| LSTM | 61.65 % | 117.50 % | 7.89e+02 | 2.142e-01 |
| LSTM_stable | 82.92 % | 152.67 % | 4.08e+03 | 3.724e-01 |
| GNN | 98.19 % | 190.23 % | 4.06e+03 | 5.972e-01 |
| GNN_stable | 103.04 % | 191.98 % | 2.74e+03 | 6.085e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 74.05 % | 458.97 % | 9.36e-01 | 4.241e-01 |
| MLP_stable | 124.74 % | 512.63 % | 6.70e+00 | 1.112e+00 |
| LSTM | 90.26 % | 390.32 % | 3.60e-01 | 6.226e-01 |
| LSTM_stable | 107.63 % | 340.11 % | 2.40e+00 | 7.879e-01 |
| GNN | 67.81 % | 332.93 % | 1.63e+00 | 3.870e-01 |
| GNN_stable | 90.42 % | 541.86 % | 2.38e+00 | 8.392e-01 |

## N = 50 takeaway

- Single-step in-distribution baseline (the headline 1-3 % number): MLP = 1.95 %, MLP_stable = 2.81 %, LSTM = 1.62 %, LSTM_stable = 3.10 %, GNN = 1.59 %, GNN_stable = 2.42 %.
- Rollout OOD stability (mean err % over 6 OOD presets): see the headline table above. The stable variant of each family is listed explicitly; compare to its single-step neighbour to read off the training-stability benefit at this N.
