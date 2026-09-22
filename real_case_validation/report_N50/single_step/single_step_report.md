# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.774e-02 | 29.63 % | 49.27 % | 1.034e+00 |
| LSTM | 3.803e-02 | 29.75 % | 49.04 % | 1.277e+00 |
| GNN | 3.808e-02 | 29.87 % | 48.99 % | 1.963e+01 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 7.377e-05 | 1.05 % | 3.69 % | 1.132e-01 |
| LSTM | 1.511e-04 | 1.39 % | 6.31 % | 1.581e-01 |
| GNN | 6.069e-05 | 1.02 % | 3.15 % | 1.117e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 5.626e-03 | 11.25 % | 20.48 % | 5.383e-02 |
| LSTM | 5.747e-03 | 11.33 % | 20.18 % | 4.819e-02 |
| GNN | 5.733e-03 | 11.44 % | 19.83 % | 4.494e-01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 4.313e-02 | 25.45 % | 52.06 % | 3.750e+00 |
| LSTM | 4.336e-02 | 25.54 % | 51.81 % | 4.162e+00 |
| GNN | 4.358e-02 | 25.84 % | 52.07 % | 9.798e+01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 7.994e-05 | 1.14 % | 3.69 % | 1.129e-01 |
| LSTM | 1.766e-04 | 1.57 % | 6.41 % | 1.513e-02 |
| GNN | 6.569e-05 | 1.10 % | 3.02 % | 6.692e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 8.327e-05 | 1.35 % | 3.67 % | 2.945e-01 |
| LSTM | 2.791e-04 | 1.45 % | 20.76 % | 2.812e-01 |
| GNN | 8.492e-06 | 0.44 % | 1.31 % | 9.629e-02 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 9.784e-10 | 0.00 % | 0.06 % | 1.155e-04 |
| LSTM | 1.780e-08 | 0.02 % | 0.34 % | 1.571e-03 |
| GNN | 8.870e-09 | 0.01 % | 0.06 % | 1.220e-03 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 0.00 % | 11.64 % |
| LSTM | 0.02 % | 11.84 % |
| GNN | 0.01 % | 11.62 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the abstract.
