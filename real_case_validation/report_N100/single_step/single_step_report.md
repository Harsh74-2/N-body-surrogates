# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.501e-01 | 77.72 % | 133.93 % | 2.475e+03 |
| MLP_stable | 5.678e-01 | 112.24 % | 249.94 % | 1.301e+03 |
| LSTM | 1.103e-01 | 51.15 % | 80.65 % | 5.774e+02 |
| LSTM_stable | 1.542e-01 | 60.42 % | 96.78 % | 1.723e+03 |
| GNN | 2.857e-01 | 80.85 % | 140.85 % | 9.222e+02 |
| GNN_stable | 2.808e-01 | 77.82 % | 140.12 % | 2.614e+03 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.936e-01 | 63.88 % | 400.65 % | 4.154e+00 |
| MLP_stable | 3.723e-01 | 73.94 % | 365.31 % | 2.552e+00 |
| LSTM | 2.829e-01 | 58.15 % | 277.58 % | 9.071e-01 |
| LSTM_stable | 6.725e-01 | 90.60 % | 370.85 % | 2.369e+00 |
| GNN | 1.914e-01 | 51.72 % | 211.01 % | 1.413e+00 |
| GNN_stable | 1.634e+00 | 136.46 % | 631.59 % | 3.496e+00 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.343e-01 | 54.29 % | 137.37 % | 5.391e+01 |
| MLP_stable | 3.440e-01 | 86.90 % | 265.08 % | 2.829e+01 |
| LSTM | 3.453e-02 | 28.66 % | 62.02 % | 1.235e+01 |
| LSTM_stable | 4.211e-02 | 33.66 % | 56.95 % | 3.757e+01 |
| GNN | 7.643e-02 | 45.38 % | 67.71 % | 1.981e+01 |
| GNN_stable | 7.174e-02 | 42.33 % | 65.14 % | 5.734e+01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.540e-01 | 80.15 % | 161.35 % | 8.742e+03 |
| MLP_stable | 8.234e-01 | 122.31 % | 295.27 % | 4.595e+03 |
| LSTM | 1.925e-01 | 57.06 % | 113.29 % | 2.041e+03 |
| LSTM_stable | 3.224e-01 | 77.08 % | 142.54 % | 6.084e+03 |
| GNN | 6.240e-01 | 107.90 % | 193.55 % | 3.441e+03 |
| GNN_stable | 6.117e-01 | 101.08 % | 193.47 % | 8.894e+03 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.325e-01 | 70.48 % | 388.78 % | 5.372e+00 |
| MLP_stable | 4.164e-01 | 81.56 % | 365.17 % | 2.906e+00 |
| LSTM | 3.237e-01 | 65.44 % | 277.38 % | 1.162e+00 |
| LSTM_stable | 7.491e-01 | 100.53 % | 375.06 % | 2.716e+00 |
| GNN | 2.046e-01 | 55.99 % | 211.01 % | 1.645e+00 |
| GNN_stable | 1.769e+00 | 151.18 % | 607.86 % | 4.412e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 6.208e-01 | 119.44 % | 376.13 % | 2.258e+00 |
| MLP_stable | 5.886e-01 | 111.70 % | 361.76 % | 1.516e+00 |
| LSTM | 8.256e-01 | 129.38 % | 343.68 % | 1.272e+00 |
| LSTM_stable | 1.113e+00 | 150.74 % | 405.38 % | 1.799e+00 |
| GNN | 7.592e-01 | 123.03 % | 297.45 % | 1.532e+00 |
| GNN_stable | 4.897e+00 | 315.58 % | 720.63 % | 2.623e+00 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.361e-04 | 2.73 % | 17.68 % | 1.197e-01 |
| MLP_stable | 4.636e-04 | 3.20 % | 20.99 % | 2.004e-01 |
| LSTM | 1.485e-04 | 1.87 % | 10.62 % | 7.530e-02 |
| LSTM_stable | 3.409e-04 | 2.75 % | 14.01 % | 1.736e-01 |
| GNN | 1.729e-04 | 1.94 % | 10.59 % | 9.342e-02 |
| GNN_stable | 2.410e-04 | 2.21 % | 10.82 % | 7.996e-02 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 2.73 % | 77.66 % |
| MLP_stable | 3.20 % | 98.11 % |
| LSTM | 1.87 % | 64.97 % |
| LSTM_stable | 2.75 % | 85.51 % |
| GNN | 1.94 % | 77.48 % |
| GNN_stable | 2.21 % | 137.41 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the abstract.
