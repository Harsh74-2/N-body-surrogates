# Real-Case Validation, N = 25

Standalone audit for the **N = 25** training-budget rerun. Two complementary modes are reported:

- **Autoregressive rollout** (in `preset_*/summary.json`) — each surrogate predicts forward from its own previous output. Errors compound; the per-step prediction becomes the *warm-up window* for the next. This is the **stress test** the user cares about: *how far can the model extrapolate before it loses the orbit?*
- **Single-step variant** (in `preset_*/ss_summary.json`) — each surrogate predicts the next frame *only*, with the warm-up window always re-built from the leapfrog reference (never from the model's own output). Errors do not compound. This is the **bare prediction error** and the headline 1-3 % number the surrogates were trained on.

## Headline (autoregressive rollout, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran, normalised by L.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 428.7 % | 274.5 % | 296.5 % |
| MLP_stable | 235.3 % | 138.4 % | 152.2 % |
| LSTM | 250.7 % | 150.8 % | 165.1 % |
| LSTM_stable | 242.5 % | 191.6 % | 198.9 % |
| GNN | 353.4 % | 137.1 % | 168.0 % |
| GNN_stable | 154.4 % | 113.5 % | 119.3 % |

## Headline (single-step, mean err %)

Each cell is the mean of `mean_err_%` over the presets that ran. The in-distribution row should sit at 1-3 % — this is the **bare** prediction error the surrogates were trained on.

| model | in-distribution | Solar-System OOD | all |
|---|---|---|---|
| MLP | 2.5 % | 92.8 % | 79.9 % |
| MLP_stable | 2.7 % | 142.7 % | 122.7 % |
| LSTM | 1.6 % | 49.0 % | 42.2 % |
| LSTM_stable | 2.6 % | 58.6 % | 50.6 % |
| GNN | 1.4 % | 74.9 % | 64.4 % |
| GNN_stable | 1.4 % | 81.7 % | 70.3 % |

## Stable vs single, per family (rollout)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 274.46 % | 138.41 % | -136.05 |
| LSTM | 150.78 % | 191.61 % | +40.83 |
| GNN | 137.07 % | 113.50 % | -23.57 |

## Stable vs single, per family (single-step)

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 92.78 % | 142.65 % | +49.87 |
| LSTM | 48.96 % | 58.60 % | +9.65 |
| GNN | 74.91 % | 81.74 % | +6.82 |

## Per-preset detail (autoregressive rollout)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 428.72 % | 1228.05 % | 36 | 4.60e+00 | 1.008e+01 |
| MLP_stable | 235.29 % | 622.47 % | 71 | 3.44e+00 | 2.515e+00 |
| LSTM | 250.70 % | 668.43 % | 66 | 2.90e+00 | 2.811e+00 |
| LSTM_stable | 242.50 % | 648.19 % | 41 | 8.13e+00 | 2.692e+00 |
| GNN | 353.44 % | 871.07 % | 95 | 4.05e+00 | 5.061e+00 |
| GNN_stable | 154.44 % | 469.44 % | 95 | 5.09e+00 | 1.065e+00 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 581.63 % | 1013.75 % | 0 | 1.65e+02 | 1.465e+01 |
| MLP_stable | 185.88 % | 422.27 % | 0 | 2.83e+01 | 1.376e+00 |
| LSTM | 241.58 % | 710.79 % | 0 | 9.86e+00 | 2.372e+00 |
| LSTM_stable | 294.95 % | 508.01 % | 0 | 2.19e+01 | 3.166e+00 |
| GNN | 248.82 % | 504.77 % | 0 | 9.38e+01 | 2.467e+00 |
| GNN_stable | 125.34 % | 394.53 % | 0 | 3.66e+01 | 6.881e-01 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 78.67 % | 204.47 % | 0 | 3.69e+01 | 2.689e-01 |
| MLP_stable | 105.21 % | 369.75 % | 0 | 7.24e+01 | 5.295e-01 |
| LSTM | 88.53 % | 205.25 % | 0 | 2.23e+01 | 3.715e-01 |
| LSTM_stable | 100.13 % | 239.22 % | 0 | 4.38e+01 | 4.688e-01 |
| GNN | 77.16 % | 202.40 % | 0 | 1.90e+01 | 2.721e-01 |
| GNN_stable | 77.83 % | 234.22 % | 0 | 2.69e+02 | 2.825e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 405.87 % | 949.19 % | 0 | 3.55e+01 | 8.718e+00 |
| MLP_stable | 188.46 % | 396.68 % | 0 | 8.78e+01 | 1.435e+00 |
| LSTM | 189.85 % | 349.92 % | 2 | 2.39e+01 | 1.396e+00 |
| LSTM_stable | 248.22 % | 453.21 % | 2 | 4.63e+01 | 2.430e+00 |
| GNN | 194.53 % | 469.90 % | 0 | 2.44e+01 | 1.465e+00 |
| GNN_stable | 123.36 % | 360.41 % | 0 | 2.82e+02 | 6.688e-01 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 311.89 % | 733.94 % | 0 | 1.27e+01 | 4.213e+00 |
| MLP_stable | 142.63 % | 611.81 % | 0 | 3.93e+01 | 9.238e-01 |
| LSTM | 178.94 % | 373.15 % | 0 | 1.04e+01 | 1.469e+00 |
| LSTM_stable | 212.38 % | 456.23 % | 0 | 1.21e+01 | 1.940e+00 |
| GNN | 132.22 % | 457.38 % | 0 | 2.28e+01 | 9.176e-01 |
| GNN_stable | 174.98 % | 619.90 % | 0 | 1.81e+01 | 1.430e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 94.19 % | 186.14 % | 0 | 3.69e+01 | 3.676e-01 |
| MLP_stable | 129.08 % | 357.26 % | 0 | 7.24e+01 | 7.755e-01 |
| LSTM | 125.65 % | 194.80 % | 0 | 2.23e+01 | 6.369e-01 |
| LSTM_stable | 142.28 % | 247.24 % | 0 | 4.38e+01 | 8.162e-01 |
| GNN | 84.52 % | 210.92 % | 0 | 2.00e+01 | 3.585e-01 |
| GNN_stable | 99.85 % | 246.78 % | 0 | 2.91e+02 | 4.474e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 174.52 % | 505.39 % | 0 | 4.34e+02 | 1.650e+00 |
| MLP_stable | 79.21 % | 422.27 % | 0 | 2.22e+01 | 3.592e-01 |
| LSTM | 80.11 % | 329.62 % | 0 | 8.30e+00 | 5.005e-01 |
| LSTM_stable | 151.67 % | 458.76 % | 0 | 2.22e+01 | 1.140e+00 |
| GNN | 85.15 % | 244.59 % | 0 | 7.03e+01 | 3.856e-01 |
| GNN_stable | 79.64 % | 272.16 % | 0 | 1.39e+01 | 3.590e-01 |

## Per-preset detail (single-step)

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity) (in-distribution)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 2.54 % | 8.14 % | 1.22e-02 | 2.446e-04 |
| MLP_stable | 2.69 % | 15.47 % | 1.96e-02 | 2.917e-04 |
| LSTM | 1.59 % | 7.34 % | 1.37e-02 | 1.094e-04 |
| LSTM_stable | 2.57 % | 10.31 % | 2.88e-02 | 2.702e-04 |
| GNN | 1.43 % | 5.24 % | 1.54e-02 | 7.846e-05 |
| GNN_stable | 1.40 % | 5.99 % | 1.21e-02 | 8.048e-05 |

### `full_solar_system` — All 8 planets + Sun (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 60.40 % | 395.10 % | 6.70e-01 | 3.279e-01 |
| MLP_stable | 107.90 % | 506.59 % | 1.67e+00 | 1.097e+00 |
| LSTM | 29.89 % | 184.02 % | 4.68e-01 | 8.207e-02 |
| LSTM_stable | 48.80 % | 234.05 % | 5.19e-01 | 1.987e-01 |
| GNN | 41.44 % | 290.99 % | 1.46e+00 | 1.602e-01 |
| GNN_stable | 52.78 % | 596.77 % | 7.39e-01 | 2.526e-01 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 73.11 % | 114.16 % | 2.80e+02 | 2.313e-01 |
| MLP_stable | 113.39 % | 202.90 % | 6.33e+02 | 6.011e-01 |
| LSTM | 53.67 % | 87.64 % | 1.13e+02 | 1.270e-01 |
| LSTM_stable | 57.69 % | 98.12 % | 2.06e+02 | 1.523e-01 |
| GNN | 77.75 % | 140.56 % | 1.04e+03 | 2.818e-01 |
| GNN_stable | 77.60 % | 141.58 % | 2.83e+01 | 2.817e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 50.04 % | 114.11 % | 5.69e+00 | 1.168e-01 |
| MLP_stable | 79.26 % | 151.55 % | 1.38e+01 | 2.740e-01 |
| LSTM | 28.05 % | 40.88 % | 2.12e+00 | 3.156e-02 |
| LSTM_stable | 30.09 % | 47.58 % | 4.21e+00 | 3.659e-02 |
| GNN | 42.27 % | 65.74 % | 2.29e+01 | 7.307e-02 |
| GNN_stable | 41.82 % | 64.29 % | 6.08e-01 | 7.142e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 232.73 % | 561.77 % | 1.34e+00 | 3.085e+00 |
| MLP_stable | 302.15 % | 650.82 % | 1.36e+00 | 4.654e+00 |
| LSTM | 85.12 % | 210.81 % | 9.62e-01 | 4.081e-01 |
| LSTM_stable | 91.86 % | 271.45 % | 1.19e+00 | 4.627e-01 |
| GNN | 144.95 % | 486.98 % | 1.50e+00 | 1.366e+00 |
| GNN_stable | 162.22 % | 871.26 % | 1.13e+00 | 1.628e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 74.39 % | 148.34 % | 9.89e+02 | 3.432e-01 |
| MLP_stable | 126.27 % | 274.13 % | 2.23e+03 | 1.013e+00 |
| LSTM | 63.64 % | 127.26 % | 4.02e+02 | 2.536e-01 |
| LSTM_stable | 70.96 % | 140.79 % | 7.28e+02 | 3.196e-01 |
| GNN | 98.46 % | 192.28 % | 4.25e+03 | 6.035e-01 |
| GNN_stable | 99.19 % | 192.61 % | 9.49e+01 | 6.079e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies) (OOD)

| model | mean err % | max err % | energy drift | MSE pos |
|---|---|---|---|---|
| MLP | 66.02 % | 354.29 % | 6.89e-01 | 3.383e-01 |
| MLP_stable | 126.96 % | 506.14 % | 1.60e+00 | 1.315e+00 |
| LSTM | 33.36 % | 184.02 % | 7.44e-01 | 9.307e-02 |
| LSTM_stable | 52.22 % | 192.07 % | 6.21e-01 | 2.037e-01 |
| GNN | 44.61 % | 248.45 % | 2.00e+00 | 1.626e-01 |
| GNN_stable | 56.83 % | 459.14 % | 1.12e+00 | 2.592e-01 |

## N = 25 takeaway

- Single-step in-distribution baseline (the headline 1-3 % number): MLP = 2.54 %, MLP_stable = 2.69 %, LSTM = 1.59 %, LSTM_stable = 2.57 %, GNN = 1.43 %, GNN_stable = 1.40 %.
- Rollout OOD stability (mean err % over 6 OOD presets): see the headline table above. The stable variant of each family is listed explicitly; compare to its single-step neighbour to read off the training-stability benefit at this N.
