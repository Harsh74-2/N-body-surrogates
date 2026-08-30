# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.253e-01 | 71.39 % | 132.89 % | 3.048e+02 |
| MLP_stable | 6.037e-01 | 120.19 % | 215.36 % | 2.793e+03 |
| LSTM | 1.114e-01 | 51.85 % | 81.85 % | 2.229e+02 |
| LSTM_stable | 1.700e-01 | 62.82 % | 103.25 % | 1.154e+03 |
| GNN | 2.795e-01 | 77.47 % | 138.81 % | 8.803e+02 |
| GNN_stable | 2.805e-01 | 78.56 % | 140.94 % | 7.735e+02 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 4.044e-01 | 69.53 % | 460.04 % | 8.438e-01 |
| MLP_stable | 9.630e-01 | 110.76 % | 513.38 % | 5.027e+00 |
| LSTM | 5.357e-01 | 79.20 % | 390.53 % | 3.038e-01 |
| LSTM_stable | 6.877e-01 | 95.42 % | 340.18 % | 1.888e+00 |
| GNN | 3.564e-01 | 60.97 % | 349.67 % | 1.341e+00 |
| GNN_stable | 7.618e-01 | 81.95 % | 550.40 % | 1.689e+00 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.336e-01 | 52.66 % | 128.77 % | 6.207e+00 |
| MLP_stable | 2.911e-01 | 84.65 % | 169.12 % | 6.128e+01 |
| LSTM | 2.895e-02 | 27.73 % | 45.25 % | 4.539e+00 |
| LSTM_stable | 4.065e-02 | 33.48 % | 61.61 % | 2.478e+01 |
| GNN | 7.127e-02 | 41.84 % | 65.33 % | 1.933e+01 |
| GNN_stable | 7.415e-02 | 43.78 % | 65.68 % | 1.633e+01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.187e-01 | 75.10 % | 150.10 % | 1.078e+03 |
| MLP_stable | 9.150e-01 | 128.96 % | 262.03 % | 9.863e+03 |
| LSTM | 2.142e-01 | 61.65 % | 117.50 % | 7.889e+02 |
| LSTM_stable | 3.724e-01 | 82.92 % | 152.67 % | 4.077e+03 |
| GNN | 5.972e-01 | 98.19 % | 190.23 % | 4.061e+03 |
| GNN_stable | 6.085e-01 | 103.04 % | 191.98 % | 2.736e+03 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 4.241e-01 | 74.05 % | 458.97 % | 9.363e-01 |
| MLP_stable | 1.112e+00 | 124.74 % | 512.63 % | 6.701e+00 |
| LSTM | 6.226e-01 | 90.26 % | 390.32 % | 3.597e-01 |
| LSTM_stable | 7.879e-01 | 107.63 % | 340.11 % | 2.403e+00 |
| GNN | 3.870e-01 | 67.81 % | 332.93 % | 1.627e+00 |
| GNN_stable | 8.392e-01 | 90.42 % | 541.86 % | 2.383e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.343e+00 | 168.03 % | 467.14 % | 1.183e+00 |
| MLP_stable | 2.863e+00 | 252.03 % | 517.33 % | 2.400e+00 |
| LSTM | 1.190e+00 | 155.22 % | 456.68 % | 1.171e+00 |
| LSTM_stable | 1.363e+00 | 170.00 % | 423.11 % | 1.634e+00 |
| GNN | 8.010e-01 | 122.53 % | 401.82 % | 1.574e+00 |
| GNN_stable | 2.080e+00 | 192.43 % | 629.98 % | 1.621e+00 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.925e-04 | 1.95 % | 12.68 % | 8.049e-02 |
| MLP_stable | 3.576e-04 | 2.81 % | 17.24 % | 1.132e-01 |
| LSTM | 1.295e-04 | 1.62 % | 11.81 % | 3.069e-02 |
| LSTM_stable | 4.568e-04 | 3.10 % | 24.41 % | 1.751e-01 |
| GNN | 1.195e-04 | 1.59 % | 9.08 % | 5.002e-02 |
| GNN_stable | 2.595e-04 | 2.42 % | 13.42 % | 5.206e-02 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 1.95 % | 85.13 % |
| MLP_stable | 2.81 % | 136.89 % |
| LSTM | 1.62 % | 77.65 % |
| LSTM_stable | 3.10 % | 92.04 % |
| GNN | 1.59 % | 78.13 % |
| GNN_stable | 2.42 % | 98.36 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the thesis abstract.
