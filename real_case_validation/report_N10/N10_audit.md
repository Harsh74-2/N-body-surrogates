# Real-Case Validation, N = 10

Standalone audit for the **N = 10** training-budget rerun. Two complementary modes are reported:

- **Autoregressive rollout** (in `preset_*/summary.json`) — each surrogate predicts forward from its own previous output. Errors compound; the per-step prediction becomes the *warm-up window* for the next. This is the **stress test** for the transfer question: *how far can the model extrapolate before it loses the orbit?*
- **Single-step variant** (in `preset_*/ss_summary.json`) — each surrogate predicts the next frame *only*, with the warm-up window always re-built from the leapfrog reference (never from the model's own output). Errors do not compound. This is the **bare prediction error** and the headline 1-3 % number the surrogates were trained on.

## Headline (autoregressive rollout, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran, normalised by L.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 286.3 % | 258.9 % | 262.8 % |
| MLP_stable | 286.8 % | 231.6 % | 239.5 % |
| LSTM | 316.8 % | 219.9 % | 233.7 % |
| LSTM_stable | 201.2 % | 185.0 % | 187.3 % |
| GNN | 176.4 % | 97.0 % | 108.4 % |
| GNN_stable | 201.2 % | 146.7 % | 154.5 % |

## Headline (single-step, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran. The in-distribution row should sit at 1-3 % — this is the **bare** prediction error the surrogates were trained on.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 3.7 % | 63.7 % | 55.1 % |
| MLP_stable | 5.7 % | 83.6 % | 72.5 % |
| LSTM | 3.7 % | 43.4 % | 37.8 % |
| LSTM_stable | 8.6 % | 78.2 % | 68.2 % |
| GNN | 3.8 % | 81.1 % | 70.1 % |
| GNN_stable | 3.4 % | 71.0 % | 61.3 % |

## Stable vs single, per family (rollout)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 258.87 % | 231.60 % | -27.26 |
| LSTM | 219.87 % | 184.98 % | -34.89 |
| GNN | 97.03 % | 146.73 % | +49.70 |

## Stable vs single, per family (single-step)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 63.70 % | 83.61 % | +19.91 |
| LSTM | 43.44 % | 78.15 % | +34.71 |
| GNN | 81.11 % | 70.97 % | -10.14 |

## Per-preset detail (autoregressive rollout)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 286.35 % | 845.31 % | 19 | 1.33e+01 | 3.959e+00 |
| MLP_stable | 286.83 % | 1085.82 % | 33 | 2.09e+00 | 4.217e+00 |
| LSTM | 316.80 % | 649.65 % | 33 | 3.59e+00 | 4.171e+00 |
| LSTM_stable | 201.17 % | 440.33 % | 19 | 6.26e+00 | 1.611e+00 |
| GNN | 176.42 % | 502.36 % | 55 | 3.20e+00 | 1.351e+00 |
| GNN_stable | 201.17 % | 603.72 % | 45 | 6.47e+00 | 1.864e+00 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 410.40 % | 818.66 % | 0 | 1.38e+01 | 7.481e+00 |
| MLP_stable | 422.84 % | 849.67 % | 0 | 2.88e+01 | 7.575e+00 |
| LSTM | 394.82 % | 661.30 % | 0 | 1.49e+03 | 5.837e+00 |
| LSTM_stable | 255.60 % | 485.51 % | 0 | 1.45e+00 | 2.553e+00 |
| GNN | 114.17 % | 402.66 % | 0 | 4.09e+02 | 5.434e-01 |
| GNN_stable | 204.57 % | 560.34 % | 0 | 1.59e+02 | 1.813e+00 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 120.70 % | 305.72 % | 0 | 1.08e+01 | 6.462e-01 |
| MLP_stable | 104.15 % | 313.33 % | 0 | 2.89e+01 | 5.277e-01 |
| LSTM | 116.61 % | 323.72 % | 0 | 5.13e+01 | 6.482e-01 |
| LSTM_stable | 123.22 % | 299.31 % | 0 | 1.39e+00 | 7.590e-01 |
| GNN | 76.67 % | 230.68 % | 0 | 7.34e+01 | 2.953e-01 |
| GNN_stable | 107.37 % | 302.10 % | 0 | 1.81e+01 | 5.425e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 384.15 % | 844.84 % | 0 | 1.06e+01 | 6.962e+00 |
| MLP_stable | 394.58 % | 892.13 % | 0 | 2.71e+01 | 7.058e+00 |
| LSTM | 398.96 % | 666.74 % | 1 | 1.02e+02 | 6.009e+00 |
| LSTM_stable | 188.17 % | 428.84 % | 1 | 1.38e+00 | 1.593e+00 |
| GNN | 156.24 % | 678.74 % | 0 | 2.14e+02 | 1.280e+00 |
| GNN_stable | 199.54 % | 520.08 % | 0 | 1.42e+02 | 1.896e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 279.62 % | 820.30 % | 0 | 7.31e+00 | 3.651e+00 |
| MLP_stable | 172.60 % | 654.42 % | 0 | 8.36e+00 | 1.337e+00 |
| LSTM | 131.92 % | 530.06 % | 0 | 2.78e+01 | 9.472e-01 |
| LSTM_stable | 200.61 % | 570.88 % | 0 | 1.27e+00 | 1.732e+00 |
| GNN | 96.09 % | 514.40 % | 0 | 1.91e+02 | 4.653e-01 |
| GNN_stable | 140.71 % | 386.26 % | 0 | 4.54e+01 | 9.826e-01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 142.59 % | 305.72 % | 0 | 1.08e+01 | 8.854e-01 |
| MLP_stable | 146.81 % | 313.34 % | 0 | 2.89e+01 | 9.263e-01 |
| LSTM | 154.59 % | 291.35 % | 0 | 5.14e+01 | 1.009e+00 |
| LSTM_stable | 193.08 % | 299.31 % | 0 | 1.39e+00 | 1.505e+00 |
| GNN | 81.13 % | 257.03 % | 0 | 2.16e+02 | 3.986e-01 |
| GNN_stable | 147.49 % | 318.08 % | 0 | 1.93e+01 | 9.414e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | max energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 215.76 % | 535.06 % | 0 | 1.31e+01 | 2.343e+00 |
| MLP_stable | 148.66 % | 798.95 % | 0 | 2.88e+01 | 1.071e+00 |
| LSTM | 122.35 % | 290.84 % | 0 | 2.05e+02 | 6.488e-01 |
| LSTM_stable | 149.20 % | 485.51 % | 0 | 1.54e+00 | 1.026e+00 |
| GNN | 57.89 % | 258.87 % | 0 | 3.07e+04 | 1.741e-01 |
| GNN_stable | 80.73 % | 292.62 % | 0 | 3.32e+01 | 3.815e-01 |

## Per-preset detail (single-step)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 3.66 % | 12.57 % | 7.89e-02 | 5.543e-04 |
| MLP_stable | 5.73 % | 21.43 % | 5.61e-02 | 1.411e-03 |
| LSTM | 3.73 % | 13.98 % | 3.76e-02 | 5.745e-04 |
| LSTM_stable | 8.64 % | 29.69 % | 6.35e-02 | 2.943e-03 |
| GNN | 3.77 % | 20.38 % | 6.34e-02 | 6.762e-04 |
| GNN_stable | 3.38 % | 21.46 % | 5.04e-02 | 5.490e-04 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 49.00 % | 323.81 % | 3.21e+00 | 2.362e-01 |
| MLP_stable | 47.40 % | 208.87 % | 1.12e+00 | 1.748e-01 |
| LSTM | 22.63 % | 116.37 % | 1.03e+00 | 3.581e-02 |
| LSTM_stable | 73.00 % | 328.41 % | 6.25e+00 | 4.170e-01 |
| GNN | 69.75 % | 490.62 % | 1.10e+00 | 4.775e-01 |
| GNN_stable | 39.65 % | 340.89 % | 1.22e+00 | 1.581e-01 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 65.19 % | 115.56 % | 1.95e+03 | 1.887e-01 |
| MLP_stable | 94.27 % | 188.69 % | 5.21e+02 | 4.080e-01 |
| LSTM | 58.10 % | 94.10 % | 5.15e+02 | 1.490e-01 |
| LSTM_stable | 67.60 % | 106.54 % | 3.78e+03 | 1.869e-01 |
| GNN | 76.17 % | 142.50 % | 2.17e+02 | 2.799e-01 |
| GNN_stable | 78.90 % | 142.78 % | 8.42e+02 | 2.865e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 41.08 % | 98.96 % | 4.25e+01 | 7.950e-02 |
| MLP_stable | 61.10 % | 145.63 % | 1.12e+01 | 1.606e-01 |
| LSTM | 32.10 % | 46.72 % | 1.09e+01 | 4.117e-02 |
| LSTM_stable | 37.93 % | 52.02 % | 8.27e+01 | 4.975e-02 |
| GNN | 40.99 % | 70.55 % | 4.92e+00 | 7.240e-02 |
| GNN_stable | 43.36 % | 67.44 % | 1.85e+01 | 7.524e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 103.85 % | 445.58 % | 2.38e+00 | 6.852e-01 |
| MLP_stable | 135.48 % | 346.45 % | 1.53e+00 | 1.065e+00 |
| LSTM | 54.06 % | 289.50 % | 1.29e+00 | 1.547e-01 |
| LSTM_stable | 122.94 % | 442.24 % | 3.10e+00 | 8.338e-01 |
| GNN | 126.24 % | 426.10 % | 1.24e+00 | 8.542e-01 |
| GNN_stable | 119.91 % | 425.60 % | 1.43e+00 | 8.959e-01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 67.82 % | 151.38 % | 6.90e+03 | 2.940e-01 |
| MLP_stable | 110.94 % | 236.89 % | 1.84e+03 | 7.392e-01 |
| LSTM | 68.89 % | 135.77 % | 1.82e+03 | 2.929e-01 |
| LSTM_stable | 87.80 % | 153.32 % | 1.33e+04 | 3.823e-01 |
| GNN | 96.95 % | 194.44 % | 1.19e+02 | 6.075e-01 |
| GNN_stable | 100.75 % | 194.14 % | 2.95e+03 | 6.142e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | mean energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 55.26 % | 305.74 % | 4.10e+00 | 2.762e-01 |
| MLP_stable | 52.45 % | 208.83 % | 1.26e+00 | 1.953e-01 |
| LSTM | 24.88 % | 116.34 % | 1.17e+00 | 4.045e-02 |
| LSTM_stable | 79.65 % | 328.32 % | 8.17e+00 | 4.673e-01 |
| GNN | 76.55 % | 434.34 % | 1.53e+00 | 5.179e-01 |
| GNN_stable | 43.27 % | 297.96 % | 1.46e+00 | 1.733e-01 |

## N = 10 takeaway

- Single-step in-distribution baseline (the headline 1-3 % number): MLP = 3.66 %, MLP_stable = 5.73 %, LSTM = 3.73 %, LSTM_stable = 8.64 %, GNN = 3.77 %, GNN_stable = 3.38 %.
- Rollout OOD stability (mean err % over 6 OOD presets): see the headline table above. The stable variant of each family is listed explicitly; compare to its single-step neighbour to read off the training-stability benefit at this N.
