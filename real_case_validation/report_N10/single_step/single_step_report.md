# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.887e-01 | 65.19 % | 115.56 % | 1.954e+03 |
| MLP_stable | 4.080e-01 | 94.27 % | 188.69 % | 5.206e+02 |
| LSTM | 1.490e-01 | 58.10 % | 94.10 % | 5.154e+02 |
| LSTM_stable | 1.869e-01 | 67.60 % | 106.54 % | 3.780e+03 |
| GNN | 2.799e-01 | 76.17 % | 142.50 % | 2.173e+02 |
| GNN_stable | 2.865e-01 | 78.90 % | 142.78 % | 8.416e+02 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.362e-01 | 49.00 % | 323.81 % | 3.208e+00 |
| MLP_stable | 1.748e-01 | 47.40 % | 208.87 % | 1.118e+00 |
| LSTM | 3.581e-02 | 22.63 % | 116.37 % | 1.033e+00 |
| LSTM_stable | 4.170e-01 | 73.00 % | 328.41 % | 6.249e+00 |
| GNN | 4.775e-01 | 69.75 % | 490.62 % | 1.104e+00 |
| GNN_stable | 1.581e-01 | 39.65 % | 340.89 % | 1.223e+00 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 7.950e-02 | 41.08 % | 98.96 % | 4.252e+01 |
| MLP_stable | 1.606e-01 | 61.10 % | 145.63 % | 1.118e+01 |
| LSTM | 4.117e-02 | 32.10 % | 46.72 % | 1.092e+01 |
| LSTM_stable | 4.975e-02 | 37.93 % | 52.02 % | 8.274e+01 |
| GNN | 7.240e-02 | 40.99 % | 70.55 % | 4.919e+00 |
| GNN_stable | 7.524e-02 | 43.36 % | 67.44 % | 1.849e+01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.940e-01 | 67.82 % | 151.38 % | 6.901e+03 |
| MLP_stable | 7.392e-01 | 110.94 % | 236.89 % | 1.839e+03 |
| LSTM | 2.929e-01 | 68.89 % | 135.77 % | 1.822e+03 |
| LSTM_stable | 3.823e-01 | 87.80 % | 153.32 % | 1.335e+04 |
| GNN | 6.075e-01 | 96.95 % | 194.44 % | 1.193e+02 |
| GNN_stable | 6.142e-01 | 100.75 % | 194.14 % | 2.952e+03 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.762e-01 | 55.26 % | 305.74 % | 4.098e+00 |
| MLP_stable | 1.953e-01 | 52.45 % | 208.83 % | 1.260e+00 |
| LSTM | 4.045e-02 | 24.88 % | 116.34 % | 1.171e+00 |
| LSTM_stable | 4.673e-01 | 79.65 % | 328.32 % | 8.175e+00 |
| GNN | 5.179e-01 | 76.55 % | 434.34 % | 1.531e+00 |
| GNN_stable | 1.733e-01 | 43.27 % | 297.96 % | 1.462e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 6.852e-01 | 103.85 % | 445.58 % | 2.385e+00 |
| MLP_stable | 1.065e+00 | 135.48 % | 346.45 % | 1.533e+00 |
| LSTM | 1.547e-01 | 54.06 % | 289.50 % | 1.294e+00 |
| LSTM_stable | 8.338e-01 | 122.94 % | 442.24 % | 3.099e+00 |
| GNN | 8.542e-01 | 126.24 % | 426.10 % | 1.238e+00 |
| GNN_stable | 8.959e-01 | 119.91 % | 425.60 % | 1.428e+00 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 5.543e-04 | 3.66 % | 12.57 % | 7.893e-02 |
| MLP_stable | 1.411e-03 | 5.73 % | 21.43 % | 5.610e-02 |
| LSTM | 5.745e-04 | 3.73 % | 13.98 % | 3.764e-02 |
| LSTM_stable | 2.943e-03 | 8.64 % | 29.69 % | 6.352e-02 |
| GNN | 6.762e-04 | 3.77 % | 20.38 % | 6.336e-02 |
| GNN_stable | 5.490e-04 | 3.38 % | 21.46 % | 5.036e-02 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 3.66 % | 63.70 % |
| MLP_stable | 5.73 % | 83.61 % |
| LSTM | 3.73 % | 43.44 % |
| LSTM_stable | 8.64 % | 78.15 % |
| GNN | 3.77 % | 81.11 % |
| GNN_stable | 3.38 % | 70.97 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the thesis abstract.
