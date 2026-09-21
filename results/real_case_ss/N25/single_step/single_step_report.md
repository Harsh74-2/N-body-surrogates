# Real-Case Validation, Single-Step Report

Same six surrogate variants (MLP / LSTM / GNN × single + stable) evaluated on real Solar-System initial conditions, but **without autoregressive rollout**. Each surrogate is asked to predict the *next frame only* from a warm-up window of `WINDOW_SIZE` leapfrog frames; the prediction is then compared directly against the leapfrog reference at that next frame. Errors do not compound because the window is always re-built from the reference, never from the model's own output.

This is the *bare* prediction error — the 1-3 % single-step MSE the surrogates were trained on. Compare with the rollout-averaged report (`real_case_report.md` in the same parent directory) to see how much the error compounds after autoregressive feedback.

## Per-preset single-step error %

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, predictions: 2495, dt_N = 2.000e-03 (in-distribution baseline)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, predictions: 2395, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, predictions: 115, dt_N = 2.886e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, predictions: 1455, dt_N = 9.419e-02 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, predictions: 115, dt_N = 5.645e-04 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, predictions: 115, dt_N = 5.127e-01 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, predictions: 115, dt_N = 3.208e-03 (out-of-distribution)

| model | MSE pos | mean err % | max err % | energy drift |
|---|---|---|---|---|

## Cross-preset aggregate (single-step mean error %)

Mean of `mean_err_%` across the presets that ran:

| model | in-distribution | Solar-System OOD |
|---|---|---|

The single-step MSE is the *honest* headline number — the rollout-averaged error in the autoregressive report grows large because errors compound over the loop. The single-step number is the one to cite in the abstract.
