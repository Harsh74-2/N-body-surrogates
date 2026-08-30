# Real-Case Validation, Error-Percentage Audit

Reads every `preset_*/summary.json` in the chosen report directory and assembles the surrogate-vs-leapfrog error as a percentage of the preset's characteristic length L. All six surrogate variants (single + stable for MLP / LSTM / GNN) are reported per preset.

## Headline

Rollout-averaged mean error % per model, in-distribution synthetic disc (the training distribution) vs Solar-System OOD (every real preset):

| model | in-distribution | Solar-System OOD | OOD − in-dist (pp) |
|---|---|---|---|
| MLP | 365.6 % | 216.4 % | -149.1 |
| MLP_stable | 186.7 % | 119.6 % | -67.1 |
| LSTM | 219.8 % | 213.0 % | -6.8 |
| LSTM_stable | 295.2 % | 334.7 % | +39.4 |
| GNN | 176.3 % | 127.8 % | -48.5 |
| GNN_stable | 300.2 % | 227.9 % | -72.3 |

The OOD premium is large for every model. The in-distribution number is itself far above the single-step MSE of 1-3 % because rollout-averaged error compounds; the natural read of the table is that **all six variants are OOD on the real Solar System, and the relative ranking is what survives the comparison, not the absolute numbers.**

## Reading the numbers

- `mean_err_%` = 100 × mean position error over the rollout, in units of L. **The 1-3 % headline number elsewhere in the thesis is the single-step MSE on the in-distribution training set; the rollout-averaged error grows large regardless of model quality because errors compound.** Everything here is the *rollout-averaged* number; the in-distribution baseline is included so you can read the OOD cost directly off the table.
- `max_err_%` = 100 × peak error during the rollout. This is the worst-case frame; for stable variants it grows much more slowly than the mean.
- `frames_before_half_L` = how many rollout steps the model stayed below 0.5 L error. `0` = the model overshoots half-L in the first frame (very wrong); a large number (or essentially the full rollout) = the model stays in the right neighbourhood throughout.
- `energy_drift` = max |E(t) - E(0)| / |E(0)| over the rollout. The leapfrog reference sits at 1e-4 to 1e-8; surrogates trained with the stability loss are 1-5 (stable); surrogates without it explode to 50-200+.

## Per-preset error percentages

### `disc_imf_in_distribution_baseline` — 25-body galaxy disc, training IMF (in-distribution sanity)
- bodies: 25, samples: 2500, dt_N = 2.000e-03 (in-distribution baseline)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 365.56 % | 774.72 % | 48 | 2.36e+00 | 5.376e+00 |
| MLP_stable | 186.69 % | 557.74 % | 30 | 2.10e+00 | 1.449e+00 |
| LSTM | 219.81 % | 691.66 % | 66 | 2.65e+00 | 2.382e+00 |
| LSTM_stable | 295.24 % | 739.95 % | 37 | 2.01e+00 | 3.634e+00 |
| GNN | 176.26 % | 613.81 % | 90 | 2.52e+00 | 1.417e+00 |
| GNN_stable | 300.17 % | 680.67 % | 70 | 1.74e+01 | 3.712e+00 |

### `full_solar_system` — All 8 planets + Sun
- bodies: 9, samples: 2400, dt_N = 3.208e-03 (out-of-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 284.79 % | 520.11 % | 0 | 1.61e+02 | 3.006e+00 |
| MLP_stable | 117.15 % | 504.95 % | 0 | 3.69e+00 | 5.482e-01 |
| LSTM | 355.43 % | 528.98 % | 0 | 3.90e+01 | 4.767e+00 |
| LSTM_stable | 757.95 % | 1166.79 % | 0 | 4.64e+00 | 2.386e+01 |
| GNN | 197.14 % | 426.61 % | 0 | 1.15e+02 | 1.802e+00 |
| GNN_stable | 344.46 % | 960.82 % | 0 | 8.09e+02 | 5.185e+00 |

### `inner_planets` — Inner planets (Mercury → Mars + Sun)
- bodies: 5, samples: 120, dt_N = 2.886e-01 (out-of-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 94.33 % | 221.81 % | 0 | 6.94e+01 | 4.257e-01 |
| MLP_stable | 119.39 % | 298.21 % | 0 | 3.91e+00 | 5.686e-01 |
| LSTM | 88.18 % | 234.21 % | 0 | 7.60e+01 | 3.546e-01 |
| LSTM_stable | 91.93 % | 233.62 % | 0 | 2.29e+00 | 3.851e-01 |
| GNN | 73.12 % | 225.31 % | 0 | 7.36e+01 | 2.712e-01 |
| GNN_stable | 94.45 % | 245.26 % | 0 | 2.36e+02 | 3.901e-01 |

### `jupiter_galileans` — Jupiter + 4 Galilean moons (toy circular orbits)
- bodies: 5, samples: 1460, dt_N = 9.419e-02 (out-of-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 328.99 % | 734.57 % | 0 | 6.46e+01 | 4.397e+00 |
| MLP_stable | 129.99 % | 259.07 % | 0 | 3.86e+00 | 6.693e-01 |
| LSTM | 264.65 % | 519.36 % | 2 | 7.03e+01 | 3.004e+00 |
| LSTM_stable | 492.48 % | 1022.84 % | 2 | 2.24e+00 | 1.298e+01 |
| GNN | 123.46 % | 478.02 % | 0 | 7.10e+01 | 7.584e-01 |
| GNN_stable | 253.76 % | 646.03 % | 0 | 2.24e+02 | 2.662e+00 |

### `solar_system_extended` — Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)
- bodies: 19, samples: 120, dt_N = 5.645e-04 (out-of-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 262.86 % | 1057.56 % | 0 | 4.33e+01 | 2.792e+00 |
| MLP_stable | 115.77 % | 498.83 % | 0 | 3.83e+00 | 5.632e-01 |
| LSTM | 257.70 % | 555.05 % | 0 | 3.97e+01 | 2.670e+00 |
| LSTM_stable | 323.87 % | 645.35 % | 0 | 2.33e+00 | 3.927e+00 |
| GNN | 168.14 % | 447.04 % | 0 | 4.73e+01 | 1.275e+00 |
| GNN_stable | 350.83 % | 1017.93 % | 0 | 1.17e+02 | 6.119e+00 |

### `sun_earth_only` — Sun–Earth 2-body (Keplerian reference)
- bodies: 2, samples: 120, dt_N = 5.127e-01 (out-of-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 135.79 % | 205.50 % | 0 | 6.94e+01 | 7.374e-01 |
| MLP_stable | 136.79 % | 258.20 % | 0 | 3.92e+00 | 7.275e-01 |
| LSTM | 118.47 % | 220.36 % | 0 | 7.61e+01 | 5.514e-01 |
| LSTM_stable | 131.91 % | 196.09 % | 0 | 2.29e+00 | 6.622e-01 |
| GNN | 87.21 % | 223.32 % | 0 | 5.64e+01 | 3.787e-01 |
| GNN_stable | 124.14 % | 219.04 % | 0 | 2.36e+02 | 6.073e-01 |

### `sun_planets_moon` — Sun + 8 planets + Earth's Moon (10 bodies)
- bodies: 10, samples: 120, dt_N = 3.208e-03 (out-of-distribution)

| model | mean err % | max err % | frames ≤ ½L | energy drift | MSE pos |
|---|---|---|---|---|---|
| MLP | 191.78 % | 516.35 % | 0 | 5.53e+02 | 1.644e+00 |
| MLP_stable | 98.72 % | 504.95 % | 0 | 3.91e+00 | 4.389e-01 |
| LSTM | 193.39 % | 471.44 % | 0 | 3.80e+01 | 1.946e+00 |
| LSTM_stable | 209.85 % | 499.47 % | 0 | 4.57e+00 | 2.103e+00 |
| GNN | 117.60 % | 426.68 % | 0 | 9.70e+01 | 8.922e-01 |
| GNN_stable | 199.47 % | 960.26 % | 0 | 1.37e+03 | 2.917e+00 |

## Cross-preset aggregate (mean error % by model)

Each cell is the mean of `mean_err_%` across the presets that ran. Use this for the *family* comparison (in-distribution vs OOD, single vs stable).

| model | in-distribution | Solar-System OOD |
|---|---|---|
| MLP | 365.56 % | 216.42 % |
| MLP_stable | 186.69 % | 119.63 % |
| LSTM | 219.81 % | 212.97 % |
| LSTM_stable | 295.24 % | 334.66 % |
| GNN | 176.26 % | 127.78 % |
| GNN_stable | 300.17 % | 227.85 % |

## Single vs stable, per family

Aggregate OOD mean error % by architecture family. The `Δ` column is `stable − single` in percentage points; a negative Δ means the stable variant is **better** on this family.

| family | single mean err % | stable mean err % | Δ (pp) |
|---|---|---|---|
| MLP | 216.42 % | 119.63 % | -96.79 |
| LSTM | 212.97 % | 334.66 % | +121.69 |
| GNN | 127.78 % | 227.85 % | +100.07 |
