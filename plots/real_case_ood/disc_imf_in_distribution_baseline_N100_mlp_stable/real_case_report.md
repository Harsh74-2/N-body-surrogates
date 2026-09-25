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
| MLP | 9.824e-01 | 6.712e-01 | 3.915e+00 | 1.503e+00 | 196 | 5.250e-05 |
