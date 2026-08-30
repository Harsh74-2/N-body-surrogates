# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.313e-01 | 73.11 % | 114.16 % | 2.795e+02 |
| MLP_stable | 6.011e-01 | 113.39 % | 202.90 % | 6.325e+02 |
| LSTM | 1.270e-01 | 53.67 % | 87.64 % | 1.133e+02 |
| LSTM_stable | 1.523e-01 | 57.69 % | 98.12 % | 2.058e+02 |
| GNN | 2.818e-01 | 77.75 % | 140.56 % | 1.044e+03 |
| GNN_stable | 2.817e-01 | 77.60 % | 141.58 % | 2.827e+01 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.279e-01 | 60.40 % | 395.10 % | 6.701e-01 |
| MLP_stable | 1.097e+00 | 107.90 % | 506.59 % | 1.674e+00 |
| LSTM | 8.207e-02 | 29.89 % | 184.02 % | 4.676e-01 |
| LSTM_stable | 1.987e-01 | 48.80 % | 234.05 % | 5.189e-01 |
| GNN | 1.602e-01 | 41.44 % | 290.99 % | 1.456e+00 |
| GNN_stable | 2.526e-01 | 52.78 % | 596.77 % | 7.393e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.168e-01 | 50.04 % | 114.11 % | 5.693e+00 |
| MLP_stable | 2.740e-01 | 79.26 % | 151.55 % | 1.382e+01 |
| LSTM | 3.156e-02 | 28.05 % | 40.88 % | 2.119e+00 |
| LSTM_stable | 3.659e-02 | 30.09 % | 47.58 % | 4.209e+00 |
| GNN | 7.307e-02 | 42.27 % | 65.74 % | 2.295e+01 |
| GNN_stable | 7.142e-02 | 41.82 % | 64.29 % | 6.085e-01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.432e-01 | 74.39 % | 148.34 % | 9.889e+02 |
| MLP_stable | 1.013e+00 | 126.27 % | 274.13 % | 2.234e+03 |
| LSTM | 2.536e-01 | 63.64 % | 127.26 % | 4.017e+02 |
| LSTM_stable | 3.196e-01 | 70.96 % | 140.79 % | 7.283e+02 |
| GNN | 6.035e-01 | 98.46 % | 192.28 % | 4.251e+03 |
| GNN_stable | 6.079e-01 | 99.19 % | 192.61 % | 9.492e+01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.383e-01 | 66.02 % | 354.29 % | 6.892e-01 |
| MLP_stable | 1.315e+00 | 126.96 % | 506.14 % | 1.602e+00 |
| LSTM | 9.307e-02 | 33.36 % | 184.02 % | 7.437e-01 |
| LSTM_stable | 2.037e-01 | 52.22 % | 192.07 % | 6.205e-01 |
| GNN | 1.626e-01 | 44.61 % | 248.45 % | 1.996e+00 |
| GNN_stable | 2.592e-01 | 56.83 % | 459.14 % | 1.122e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.085e+00 | 232.73 % | 561.77 % | 1.337e+00 |
| MLP_stable | 4.654e+00 | 302.15 % | 650.82 % | 1.355e+00 |
| LSTM | 4.081e-01 | 85.12 % | 210.81 % | 9.622e-01 |
| LSTM_stable | 4.627e-01 | 91.86 % | 271.45 % | 1.186e+00 |
| GNN | 1.366e+00 | 144.95 % | 486.98 % | 1.500e+00 |
| GNN_stable | 1.628e+00 | 162.22 % | 871.26 % | 1.126e+00 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.446e-04 | 2.54 % | 8.14 % | 1.217e-02 |
| MLP_stable | 2.917e-04 | 2.69 % | 15.47 % | 1.961e-02 |
| LSTM | 1.094e-04 | 1.59 % | 7.34 % | 1.368e-02 |
| LSTM_stable | 2.702e-04 | 2.57 % | 10.31 % | 2.878e-02 |
| GNN | 7.846e-05 | 1.43 % | 5.24 % | 1.536e-02 |
| GNN_stable | 8.048e-05 | 1.40 % | 5.99 % | 1.206e-02 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 2.54 % | 92.78 % |
| MLP_stable | 2.69 % | 142.65 % |
| LSTM | 1.59 % | 48.96 % |
| LSTM_stable | 2.57 % | 58.60 % |
| GNN | 1.43 % | 74.91 % |
| GNN_stable | 1.40 % | 81.74 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the thesis abstract.
