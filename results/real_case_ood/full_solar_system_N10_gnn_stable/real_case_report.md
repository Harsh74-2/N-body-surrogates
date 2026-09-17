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

### full_solar_system, All 8 planets + Sun

- N = 9, samples = 2400, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| GNN | 1.999e-01 | 6.451e+00 | 2.283e+00 | 6.005e-01 | 93 | 3.732e-04 |

## Kepler's 3rd-law check (reference integrator)

For each preset we measure the orbital period T and semi-major axis a of every non-primary body from the reference trajectory, and compare T²/a³ to the predicted 4π²/(G·M_primary). All bodies in the same preset should give the same K = T²/a³ (that's the law). The deviation is reported as a percentage. Bodies that don't complete at least one full orbit in the simulation window show NaN: increase `duration_years` to bring them in.

### full_solar_system, Kepler 3rd-law check

Reference integrator (high-precision leapfrog, dt_N = 3.208e-03, 2400 samples over 200 yr). For each non-primary body we report the orbital period T (median full-revolution spacing, axis-crossing detector) and the semi-major axis a = (r_min + r_max)/2, then the Kepler ratio K = T²/a³. The predicted K = 4π²/(G·M_primary) is the same for every body in this frame. Bodies whose orbit does not complete at least one full period in the simulation window show NaN, increase `duration_years` to bring them in.

| body | is_primary | mass (kg) | T (yr) | a (AU) | T²/a³ | K_pred | deviation (%) |
|---|---|---|---|---|---|---|---|
| Sun | ✓ | 1.988e+30 | — | — | — | 3.9531e+01 | — |
| Mercury |   | 3.301e+23 | 0.2441 | 0.3872 | 4.0568e+01 | 3.9531e+01 | +2.6224 |
| Venus |   | 4.868e+24 | 0.615 | 0.7233 | 3.9501e+01 | 3.9531e+01 | -0.0763 |
| Earth |   | 5.972e+24 | 1 | 1 | 3.9533e+01 | 3.9531e+01 | +0.0031 |
| Mars |   | 6.417e+23 | 1.881 | 1.524 | 3.9532e+01 | 3.9531e+01 | +0.0022 |
| Jupiter |   | 1.898e+27 | 11.86 | 5.201 | 3.9509e+01 | 3.9531e+01 | -0.0554 |
| Saturn |   | 5.683e+26 | 29.47 | 9.544 | 3.9499e+01 | 3.9531e+01 | -0.0820 |
| Uranus |   | 8.681e+25 | 84 | 19.19 | 3.9463e+01 | 3.9531e+01 | -0.1730 |
| Neptune |   | 1.024e+26 | - | 30.02 | - | 3.9531e+01 | - |

_The shortest-period body in this preset completes ≈ 819 full orbits in the simulation window; the longest-period body shown above completes fewer, so its T estimate is noisier._
