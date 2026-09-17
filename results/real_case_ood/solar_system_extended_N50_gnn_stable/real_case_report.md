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

### solar_system_extended, Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)

- N = 19, samples = 120, dt_N = 5.645e-04
- scale: M = 1.991e+30 kg, L = 1.423e+13 m, T = 4.658e+09 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| GNN | 1.396e-02 | 1.247e+01 | 7.556e-01 | 1.259e-01 | 75 | 2.459e-06 |

## Kepler's 3rd-law check (reference integrator)

For each preset we measure the orbital period T and semi-major axis a of every non-primary body from the reference trajectory, and compare T²/a³ to the predicted 4π²/(G·M_primary). All bodies in the same preset should give the same K = T²/a³ (that's the law). The deviation is reported as a percentage. Bodies that don't complete at least one full orbit in the simulation window show NaN: increase `duration_years` to bring them in.

### solar_system_extended, Kepler 3rd-law check

Reference integrator (high-precision leapfrog, dt_N = 5.645e-04, 120 samples over 10 yr). For each non-primary body we report the orbital period T (median full-revolution spacing, axis-crossing detector) and the semi-major axis a = (r_min + r_max)/2, then the Kepler ratio K = T²/a³. The predicted K = 4π²/(G·M_primary) is the same for every body in this frame. Bodies whose orbit does not complete at least one full period in the simulation window show NaN, increase `duration_years` to bring them in.

| body | is_primary | mass (kg) | T (yr) | a (AU) | T²/a³ | K_pred | deviation (%) |
|---|---|---|---|---|---|---|---|
| Sun | ✓ | 1.988e+30 | — | — | — | 3.9531e+01 | — |
| Mercury |   | 3.301e+23 | 0.2443 | 0.3877 | 4.0458e+01 | 3.9531e+01 | +2.3432 |
| Venus |   | 4.868e+24 | 0.6151 | 0.7235 | 3.9492e+01 | 3.9531e+01 | -0.1008 |
| Earth |   | 5.972e+24 | 1.001 | 1 | 3.9540e+01 | 3.9531e+01 | +0.0208 |
| Mars |   | 6.417e+23 | 1.881 | 1.524 | 3.9534e+01 | 3.9531e+01 | +0.0078 |
| Jupiter |   | 1.898e+27 | - | 5.216 | - | 3.9531e+01 | - |
| Saturn |   | 5.683e+26 | - | 9.138 | - | 3.9531e+01 | - |
| Uranus |   | 8.681e+25 | - | 19.12 | - | 3.9531e+01 | - |
| Neptune |   | 1.024e+26 | - | 29.85 | - | 3.9531e+01 | - |
| Moon |   | 7.342e+22 | 0.9659 | 0.9773 | 3.9502e+01 | 3.9531e+01 | -0.0738 |
| Pluto |   | 1.303e+22 | - | 49.66 | - | 3.9531e+01 | - |
| Eris |   | 1.660e+22 | - | 66.87 | - | 3.9531e+01 | - |
| Ceres |   | 9.393e+20 | - | 37 | - | 3.9531e+01 | - |
| Makemake |   | 3.100e+21 | - | 80.55 | - | 3.9531e+01 | - |
| Haumea |   | 4.006e+21 | - | 78.35 | - | 3.9531e+01 | - |
| Jupiter-Io |   | 8.932e+22 | - | 20.11 | - | 3.9531e+01 | - |
| Jupiter-Europa |   | 4.800e+22 | - | 4.575 | - | 3.9531e+01 | - |
| Jupiter-Ganymede |   | 1.482e+23 | - | 5.215 | - | 3.9531e+01 | - |
| Jupiter-Callisto |   | 1.076e+23 | - | 5.216 | - | 3.9531e+01 | - |

_The shortest-period body in this preset completes ≈ 40 full orbits in the simulation window; the longest-period body shown above completes fewer, so its T estimate is noisier._
