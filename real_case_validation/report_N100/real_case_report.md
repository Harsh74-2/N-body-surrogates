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
| GNN | 4.039e+00 | 2.280e+00 | 8.952e+00 | 2.959e+00 | 65 | 2.111e+00 |
| GNN_stable | 2.501e+00 | 1.568e+00 | 6.919e+00 | 2.342e+00 | 115 | 1.454e+00 |
| LSTM | 2.104e+00 | 1.284e+00 | 5.519e+00 | 2.223e+00 | 68 | 8.836e+00 |
| LSTM_stable | 2.444e+00 | 1.407e+00 | 7.443e+00 | 2.322e+00 | 40 | 8.933e+00 |
| MLP | 2.286e+00 | 1.325e+00 | 6.210e+00 | 2.278e+00 | 30 | 4.800e+00 |
| MLP_stable | 1.349e+00 | 9.268e-01 | 4.786e+00 | 1.831e+00 | 23 | 4.297e+00 |
