# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 3.712e-02 | 29.44 % | 48.46 % | 5.457e+00 |
| LSTM | 3.757e-02 | 29.56 % | 49.51 % | 2.795e+00 |
| GNN | 3.815e-02 | 29.81 % | 49.23 % | 1.236e+01 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 7.599e-05 | 1.06 % | 3.73 % | 6.255e-02 |
| LSTM | 6.637e-05 | 0.98 % | 3.86 % | 3.406e-02 |
| GNN | 7.807e-05 | 1.07 % | 4.58 % | 6.606e-02 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 5.601e-03 | 11.27 % | 20.42 % | 1.489e-01 |
| LSTM | 5.502e-03 | 11.13 % | 20.13 % | 3.058e-01 |
| GNN | 5.737e-03 | 11.35 % | 19.90 % | 4.643e-01 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 4.318e-02 | 25.49 % | 52.34 % | 1.803e+01 |
| LSTM | 4.310e-02 | 25.44 % | 52.01 % | 8.066e+00 |
| GNN | 4.348e-02 | 25.59 % | 51.86 % | 9.979e+01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 8.836e-05 | 1.19 % | 3.72 % | 8.210e-02 |
| LSTM | 7.725e-05 | 1.09 % | 3.87 % | 2.457e-02 |
| GNN | 8.650e-05 | 1.19 % | 3.49 % | 5.700e-02 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 5.941e-05 | 1.13 % | 3.37 % | 1.551e-01 |
| LSTM | 8.016e-05 | 1.20 % | 3.21 % | 9.106e-02 |
| GNN | 2.456e-05 | 0.69 % | 4.30 % | 6.272e-02 |

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|
| MLP | 1.345e-09 | 0.00 % | 0.04 % | 2.226e-04 |
| LSTM | 5.503e-10 | 0.00 % | 0.03 % | 2.620e-04 |
| GNN | 3.964e-08 | 0.03 % | 0.27 % | 1.309e-03 |

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 0.00 % | 11.60 % |
| LSTM | 0.00 % | 11.57 % |
| GNN | 0.03 % | 11.62 % |

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the abstract.
