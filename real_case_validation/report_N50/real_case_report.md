# Real-Case Validation Report

Trained MLP / LSTM / GNN surrogates evaluated on real Solar-System initial conditions. All numbers are in the dimensionless N-body units the surrogates were trained on.

## Out-of-distribution caveat

The surrogates were trained on 25-body synthetic galaxy discs (`simulation_3d.init_galaxy_disc`, mass ratio ≲ 10, body count = 25, Σm = 1, G = 1, no central sink). The Solar System is a *very* different distribution: 8-10 bodies with mass ratios of 10⁵ (Sun:Earth) or higher. The numbers below therefore measure **out-of-distribution generalisation**, not domain fit. The `disc_imf_in_distribution_baseline` preset provides an in-distribution sanity check for comparison.

## Per-preset summary

### Reading key

Every line in the plots uses one of the styles below. References are drawn in white. The book (closed-form Kepler) line is green. Surrogates use a different colour and linestyle per architecture:

| line        | colour   | linestyle | meaning |
|-------------|----------|-----------|---------|
| book        | green    | solid     | Closed-form 2-body Kepler (primary + body, all other perturbations ignored) |
| reference   | white    | solid     | Leapfrog at dt_ref = coarse dt / 100 |
| GNN         | blue     | solid     | Trained GNN surrogate (`model_best.pt`) |
| GNN_stable  | blue     | dashed    | GNN trained with stability loss (`model_best.pt` from `*/gnn_stable/`) |
| LSTM        | orange   | dash-dot  | Trained LSTM surrogate (`model_best.pt` from `*/lstm/`) |
| LSTM_stable | orange   | dotted    | LSTM trained with stability loss (`model_best.pt` from `*/lstm_stable/`) |
| MLP         | violet   | dotted    | Trained MLP surrogate (`model_best.pt` from `*/mlp/`) |
| MLP_stable  | violet   | densely dotted | MLP trained with stability loss (`model_best.pt` from `*/mlp_stable/`) |

### disc_imf_in_distribution_baseline, 25-body galaxy disc, training IMF (in-distribution sanity)

- N = 25, samples = 2500, dt_N = 2.000e-03

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| GNN | 1.417e+00 | 1.048e+00 | 6.138e+00 | 1.763e+00 | 90 | 2.523e+00 |
| GNN_stable | 3.712e+00 | 2.015e+00 | 6.807e+00 | 3.002e+00 | 70 | 1.736e+01 |
| LSTM | 2.382e+00 | 1.397e+00 | 6.917e+00 | 2.198e+00 | 66 | 2.650e+00 |
| LSTM_stable | 3.634e+00 | 1.940e+00 | 7.400e+00 | 2.952e+00 | 37 | 2.014e+00 |
| MLP | 5.376e+00 | 2.922e+00 | 7.747e+00 | 3.656e+00 | 48 | 2.362e+00 |
| MLP_stable | 1.449e+00 | 1.059e+00 | 5.577e+00 | 1.867e+00 | 30 | 2.098e+00 |
