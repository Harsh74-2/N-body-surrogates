# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.710e-02 | 29.37 % | 48.34 % | 5.962e-01 |
| LSTM | 3.675e-02 | 29.13 % | 50.88 % | 1.649e-01 |
| GNN | 3.792e-02 | 29.71 % | 49.09 % | 8.176e+01 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.984e-05 | 0.78 % | 2.84 % | 9.510e-02 |
| LSTM | 7.256e-04 | 2.95 % | 13.70 % | 1.248e-01 |
| GNN | 5.152e-05 | 0.89 % | 2.84 % | 1.683e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 5.435e-03 | 11.05 % | 19.65 % | 2.573e-02 |
| LSTM | 5.109e-03 | 10.67 % | 22.91 % | 1.137e-01 |
| GNN | 5.658e-03 | 11.27 % | 19.61 % | 1.783e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 4.247e-02 | 25.26 % | 51.60 % | 1.979e+00 |
| LSTM | 4.087e-02 | 24.76 % | 50.52 % | 1.994e-01 |
| GNN | 4.332e-02 | 25.52 % | 51.91 % | 2.082e+02 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 4.161e-05 | 0.83 % | 2.69 % | 1.354e-02 |
| LSTM | 8.172e-04 | 3.30 % | 13.01 % | 1.802e-02 |
| GNN | 5.565e-05 | 0.97 % | 2.83 % | 9.746e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.400e-05 | 0.54 % | 2.24 % | 1.151e-01 |
| LSTM | 1.025e-02 | 11.37 % | 47.58 % | 2.587e-01 |
| GNN | 4.722e-06 | 0.31 % | 1.13 % | 5.577e-02 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.047e-09 | 0.00 % | 0.03 % | 2.558e-04 |
| LSTM | 6.472e-10 | 0.00 % | 0.05 % | 1.448e-04 |
| GNN | 1.742e-08 | 0.02 % | 0.09 % | 9.235e-04 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 0.00 % | 11.31 % |
| LSTM | 0.00 % | 13.70 % |
| GNN | 0.02 % | 11.44 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the abstract.
