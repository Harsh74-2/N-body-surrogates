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
| GNN | 5.061e+00 | 2.796e+00 | 8.711e+00 | 3.534e+00 | 95 | 4.051e+00 |
| GNN_stable | 1.065e+00 | 1.024e+00 | 4.694e+00 | 1.544e+00 | 95 | 5.093e+00 |
| LSTM | 2.811e+00 | 1.586e+00 | 6.684e+00 | 2.507e+00 | 66 | 2.902e+00 |
| LSTM_stable | 2.692e+00 | 1.474e+00 | 6.482e+00 | 2.425e+00 | 41 | 8.134e+00 |
| MLP | 1.008e+01 | 5.369e+00 | 1.228e+01 | 4.287e+00 | 36 | 4.596e+00 |
| MLP_stable | 2.515e+00 | 1.424e+00 | 6.225e+00 | 2.353e+00 | 71 | 3.436e+00 |
