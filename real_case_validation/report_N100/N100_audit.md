# Real-Case Validation, N = 100

Standalone audit for the **N = 100** training-budget rerun. Two complementary modes are reported:

- **Autoregressive rollout** (in `preset_*/summary.json`) — each surrogate predicts forward from its own previous output. Errors compound; the per-step prediction becomes the *warm-up window* for the next. This is the **stress test** for the transfer question: *how far can the model extrapolate before it loses the orbit?*
- **Single-step variant** (in `preset_*/ss_summary.json`) — each surrogate predicts the next frame *only*, with the warm-up window always re-built from the leapfrog reference (never from the model's own output). Errors do not compound. This is the **bare prediction error** and the headline 1-3 % number the surrogates were trained on.

## Headline (autoregressive rollout, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran, normalised by L.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 227.8 % | 191.2 % | 196.4 % |
| MLP_stable | 183.1 % | 105.6 % | 116.6 % |
| LSTM | 222.3 % | 148.0 % | 158.6 % |
| LSTM_stable | 232.2 % | 170.5 % | 179.3 % |
| GNN | 295.9 % | 196.5 % | 210.7 % |
| GNN_stable | 234.2 % | 217.1 % | 219.5 % |

## Headline (single-step, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran. The in-distribution row should sit at 1-3 % — this is the **bare** prediction error the surrogates were trained on.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 2.7 % | 77.7 % | 67.0 % |
| MLP_stable | 3.2 % | 98.1 % | 84.6 % |
| LSTM | 1.9 % | 65.0 % | 56.0 % |
| LSTM_stable | 2.8 % | 85.5 % | 73.7 % |
| GNN | 1.9 % | 77.5 % | 66.7 % |
| GNN_stable | 2.2 % | 137.4 % | 118.1 % |

## Stable vs single, per family (rollout)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 191.19 % | 105.57 % | -85.62 |
| LSTM | 147.97 % | 170.50 % | +22.53 |
| GNN | 196.50 % | 217.06 % | +20.56 |

## Stable vs single, per family (single-step)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 77.66 % | 98.11 % | +20.45 |
| LSTM | 64.97 % | 85.51 % | +20.53 |
| GNN | 77.48 % | 137.41 % | +59.93 |

## Per-preset detail (autoregressive rollout)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 227.79 % | 621.02 % | 30 | 4.80e+00 | 2.286e+00 |
| MLP_stable | 183.14 % | 478.64 % | 23 | 4.30e+00 | 1.349e+00 |
| LSTM | 222.30 % | 551.89 % | 68 | 8.84e+00 | 2.104e+00 |
| LSTM_stable | 232.16 % | 744.33 % | 40 | 8.93e+00 | 2.444e+00 |
| GNN | 295.93 % | 895.16 % | 65 | 2.11e+00 | 4.039e+00 |
| GNN_stable | 234.19 % | 691.89 % | 115 | 1.45e+00 | 2.501e+00 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 267.44 % | 731.01 % | 0 | 5.01e+01 | 2.959e+00 |
| MLP_stable | 99.26 % | 232.39 % | 0 | 2.11e+01 | 3.940e-01 |
| LSTM | 158.26 % | 331.18 % | 0 | 6.66e+01 | 9.524e-01 |
| LSTM_stable | 288.20 % | 749.49 % | 0 | 1.68e+01 | 4.148e+00 |
| GNN | 368.38 % | 588.90 % | 0 | 2.63e+01 | 5.472e+00 |
| GNN_stable | 290.79 % | 905.56 % | 0 | 1.96e+01 | 3.853e+00 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 110.06 % | 234.70 % | 0 | 3.81e+01 | 5.116e-01 |
| MLP_stable | 108.82 % | 240.08 % | 0 | 2.12e+01 | 4.744e-01 |
| LSTM | 94.04 % | 219.60 % | 0 | 1.13e+02 | 4.169e-01 |
| LSTM_stable | 109.02 % | 252.41 % | 0 | 1.10e+01 | 5.089e-01 |
| GNN | 86.77 % | 249.14 % | 0 | 9.24e+00 | 3.304e-01 |
| GNN_stable | 78.78 % | 234.22 % | 0 | 1.04e+01 | 2.959e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 280.52 % | 742.58 % | 0 | 3.90e+01 | 3.679e+00 |
| MLP_stable | 111.88 % | 250.89 % | 0 | 2.08e+01 | 4.908e-01 |
| LSTM | 147.75 % | 259.88 % | 2 | 1.08e+02 | 7.792e-01 |
| LSTM_stable | 166.70 % | 328.01 % | 0 | 1.08e+01 | 1.070e+00 |
| GNN | 298.79 % | 661.86 % | 0 | 8.65e+00 | 4.215e+00 |
| GNN_stable | 190.39 % | 471.72 % | 0 | 1.03e+01 | 1.680e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 224.78 % | 721.51 % | 0 | 3.66e+01 | 2.053e+00 |
| MLP_stable | 95.88 % | 460.25 % | 0 | 2.31e+01 | 4.203e-01 |
| LSTM | 230.69 % | 517.29 % | 0 | 7.58e+01 | 2.113e+00 |
| LSTM_stable | 191.12 % | 479.38 % | 0 | 1.05e+01 | 1.530e+00 |
| GNN | 204.41 % | 586.73 % | 0 | 7.39e+00 | 2.171e+00 |
| GNN_stable | 427.92 % | 1044.58 % | 0 | 8.44e+00 | 8.173e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 127.72 % | 211.94 % | 0 | 3.81e+01 | 6.190e-01 |
| MLP_stable | 133.22 % | 219.22 % | 0 | 2.12e+01 | 6.641e-01 |
| LSTM | 132.63 % | 211.66 % | 0 | 1.13e+02 | 7.093e-01 |
| LSTM_stable | 144.69 % | 221.03 % | 0 | 1.10e+01 | 7.947e-01 |
| GNN | 112.65 % | 256.71 % | 0 | 1.03e+01 | 5.311e-01 |
| GNN_stable | 94.76 % | 231.34 % | 0 | 1.09e+01 | 4.349e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 136.60 % | 354.23 % | 0 | 4.62e+01 | 8.465e-01 |
| MLP_stable | 84.35 % | 232.39 % | 0 | 1.96e+01 | 3.144e-01 |
| LSTM | 124.45 % | 246.14 % | 0 | 6.67e+01 | 6.897e-01 |
| LSTM_stable | 123.27 % | 510.23 % | 0 | 1.68e+01 | 8.536e-01 |
| GNN | 107.97 % | 482.32 % | 0 | 3.58e+01 | 6.888e-01 |
| GNN_stable | 219.73 % | 698.97 % | 0 | 1.81e+01 | 3.021e+00 |

## Per-preset detail (single-step)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 2.73 % | 17.68 % | 1.20e-01 | 3.361e-04 |
| MLP_stable | 3.20 % | 20.99 % | 2.00e-01 | 4.636e-04 |
| LSTM | 1.87 % | 10.62 % | 7.53e-02 | 1.485e-04 |
| LSTM_stable | 2.75 % | 14.01 % | 1.74e-01 | 3.409e-04 |
| GNN | 1.94 % | 10.59 % | 9.34e-02 | 1.729e-04 |
| GNN_stable | 2.21 % | 10.82 % | 8.00e-02 | 2.410e-04 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 63.88 % | 400.65 % | 4.15e+00 | 2.936e-01 |
| MLP_stable | 73.94 % | 365.31 % | 2.55e+00 | 3.723e-01 |
| LSTM | 58.15 % | 277.58 % | 9.07e-01 | 2.829e-01 |
| LSTM_stable | 90.60 % | 370.85 % | 2.37e+00 | 6.725e-01 |
| GNN | 51.72 % | 211.01 % | 1.41e+00 | 1.914e-01 |
| GNN_stable | 136.46 % | 631.59 % | 3.50e+00 | 1.634e+00 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 77.72 % | 133.93 % | 2.47e+03 | 2.501e-01 |
| MLP_stable | 112.24 % | 249.94 % | 1.30e+03 | 5.678e-01 |
| LSTM | 51.15 % | 80.65 % | 5.77e+02 | 1.103e-01 |
| LSTM_stable | 60.42 % | 96.78 % | 1.72e+03 | 1.542e-01 |
| GNN | 80.85 % | 140.85 % | 9.22e+02 | 2.857e-01 |
| GNN_stable | 77.82 % | 140.12 % | 2.61e+03 | 2.808e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 54.29 % | 137.37 % | 5.39e+01 | 1.343e-01 |
| MLP_stable | 86.90 % | 265.08 % | 2.83e+01 | 3.440e-01 |
| LSTM | 28.66 % | 62.02 % | 1.24e+01 | 3.453e-02 |
| LSTM_stable | 33.66 % | 56.95 % | 3.76e+01 | 4.211e-02 |
| GNN | 45.38 % | 67.71 % | 1.98e+01 | 7.643e-02 |
| GNN_stable | 42.33 % | 65.14 % | 5.73e+01 | 7.174e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 119.44 % | 376.13 % | 2.26e+00 | 6.208e-01 |
| MLP_stable | 111.70 % | 361.76 % | 1.52e+00 | 5.886e-01 |
| LSTM | 129.38 % | 343.68 % | 1.27e+00 | 8.256e-01 |
| LSTM_stable | 150.74 % | 405.38 % | 1.80e+00 | 1.113e+00 |
| GNN | 123.03 % | 297.45 % | 1.53e+00 | 7.592e-01 |
| GNN_stable | 315.58 % | 720.63 % | 2.62e+00 | 4.897e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 80.15 % | 161.35 % | 8.74e+03 | 3.540e-01 |
| MLP_stable | 122.31 % | 295.27 % | 4.60e+03 | 8.234e-01 |
| LSTM | 57.06 % | 113.29 % | 2.04e+03 | 1.925e-01 |
| LSTM_stable | 77.08 % | 142.54 % | 6.08e+03 | 3.224e-01 |
| GNN | 107.90 % | 193.55 % | 3.44e+03 | 6.240e-01 |
| GNN_stable | 101.08 % | 193.47 % | 8.89e+03 | 6.117e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 70.48 % | 388.78 % | 5.37e+00 | 3.325e-01 |
| MLP_stable | 81.56 % | 365.17 % | 2.91e+00 | 4.164e-01 |
| LSTM | 65.44 % | 277.38 % | 1.16e+00 | 3.237e-01 |
| LSTM_stable | 100.53 % | 375.06 % | 2.72e+00 | 7.491e-01 |
| GNN | 55.99 % | 211.01 % | 1.64e+00 | 2.046e-01 |
| GNN_stable | 151.18 % | 607.86 % | 4.41e+00 | 1.769e+00 |

## N = 100 takeaway

- Single-step in-distribution baseline (the headline 1-3 % number): MLP = 2.73 %, MLP_stable = 3.20 %, LSTM = 1.87 %, LSTM_stable = 2.75 %, GNN = 1.94 %, GNN_stable = 2.21 %.
- Rollout OOD stability (mean err % over 6 OOD presets): see the headline table above. The stable variant of each family is listed explicitly; compare to its single-step neighbour to read off the training-stability benefit at this N.
