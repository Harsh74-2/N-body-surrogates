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

### jupiter_galileans, Jupiter + 4 Galilean moons (toy circular orbits)

- N = 5, samples = 1460, dt_N = 9.419e-02
- scale: M = 1.899e+27 kg, L = 1.883e+09 m, T = 2.295e+05 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| LSTM | 2.455e-01 | 7.925e-01 | 2.264e+00 | 6.458e-01 | 3 | 2.004e-03 |

## Kepler's 3rd-law check (reference integrator)

For each preset we measure the orbital period T and semi-major axis a of every non-primary body from the reference trajectory, and compare T²/a³ to the predicted 4π²/(G·M_primary). All bodies in the same preset should give the same K = T²/a³ (that's the law). The deviation is reported as a percentage. Bodies that don't complete at least one full orbit in the simulation window show NaN: increase `duration_years` to bring them in.

### jupiter_galileans, Kepler 3rd-law check

Reference integrator (high-precision leapfrog, dt_N = 9.419e-02, 1460 samples over 1 yr). For each non-primary body we report the orbital period T (median full-revolution spacing, axis-crossing detector) and the semi-major axis a = (r_min + r_max)/2, then the Kepler ratio K = T²/a³. The predicted K = 4π²/(G·M_primary) is the same for every body in this frame. Bodies whose orbit does not complete at least one full period in the simulation window show NaN, increase `duration_years` to bring them in.

| body | is_primary | mass (kg) | T (yr) | a (AU) | T²/a³ | K_pred | deviation (%) |
|---|---|---|---|---|---|---|---|
| Jupiter | ✓ | 1.898e+27 | — | — | — | 3.9487e+01 | — |
| Io |   | 8.932e+22 | 0.004844 | 0.002819 | 3.9474e+01 | 3.9487e+01 | -0.0326 |
| Europa |   | 4.800e+22 | 0.009718 | 0.004484 | 3.9480e+01 | 3.9487e+01 | -0.0162 |
| Ganymede |   | 1.482e+23 | 0.01958 | 0.007152 | 3.9477e+01 | 3.9487e+01 | -0.0247 |
| Callisto |   | 1.076e+23 | 0.04564 | 0.01258 | 3.9473e+01 | 3.9487e+01 | -0.0340 |

_The shortest-period body in this preset completes ≈ 206 full orbits in the simulation window; the longest-period body shown above completes fewer, so its T estimate is noisier._
