# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.747e-02 | 29.51 % | 48.18 % | 6.298e+00 |
| LSTM | 3.754e-02 | 29.40 % | 51.44 % | 6.412e-01 |
| GNN | 3.828e-02 | 29.85 % | 49.37 % | 3.047e+01 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 6.877e-05 | 1.00 % | 4.11 % | 1.138e-01 |
| LSTM | 1.173e-03 | 3.34 % | 19.96 % | 1.259e-01 |
| GNN | 7.873e-05 | 1.08 % | 3.58 % | 1.337e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 5.518e-03 | 11.10 % | 21.02 % | 5.463e-02 |
| LSTM | 5.262e-03 | 10.76 % | 23.30 % | 3.472e-02 |
| GNN | 5.795e-03 | 11.39 % | 20.16 % | 7.505e-01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 4.267e-02 | 25.31 % | 51.60 % | 2.129e+01 |
| LSTM | 4.057e-02 | 24.67 % | 50.64 % | 2.253e+00 |
| GNN | 4.361e-02 | 25.63 % | 51.99 % | 1.016e+02 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 7.363e-05 | 1.07 % | 3.96 % | 6.536e-02 |
| LSTM | 1.387e-03 | 3.82 % | 19.95 % | 2.059e-02 |
| GNN | 8.717e-05 | 1.20 % | 3.37 % | 5.478e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 2.162e-04 | 1.97 % | 7.93 % | 2.626e-01 |
| LSTM | 5.876e-03 | 9.70 % | 23.69 % | 2.547e-01 |
| GNN | 5.155e-05 | 0.63 % | 24.24 % | 5.024e-02 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 6.160e-10 | 0.00 % | 0.04 % | 1.173e-04 |
| LSTM | 3.177e-10 | 0.00 % | 0.03 % | 7.937e-05 |
| GNN | 3.162e-08 | 0.03 % | 0.14 % | 7.309e-04 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 0.00 % | 11.66 % |
| LSTM | 0.00 % | 13.61 % |
| GNN | 0.03 % | 11.63 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the abstract.
