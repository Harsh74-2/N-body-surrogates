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

### inner_planets, Inner planets (Mercury → Mars + Sun)

- N = 5, samples = 120, dt_N = 2.886e-01
- scale: M = 1.988e+30 kg, L = 2.225e+11 m, T = 9.111e+06 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 2.621e-01 | 6.312e-01 | 2.152e+00 | 6.798e-01 | 1 | 4.525e+04 |
| LSTM | 3.333e-01 | 3.224e+00 | 2.439e+00 | 7.795e-01 | 1 | 9.278e+05 |
| GNN | 2.146e-01 | 8.755e-01 | 2.041e+00 | 5.999e-01 | 1 | 2.788e+03 |
| MLP_stable | 2.604e-01 | 6.030e-01 | 2.155e+00 | 6.757e-01 | 1 | 4.600e+04 |
| LSTM_stable | 2.213e-01 | 9.754e-01 | 2.035e+00 | 6.491e-01 | 1 | 4.450e+05 |
| GNN_stable | 2.165e-01 | 8.268e-01 | 2.042e+00 | 6.020e-01 | 1 | 6.506e+03 |

### full_solar_system, All 8 planets + Sun

- N = 9, samples = 2400, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 9.958e+00 | 1.174e+01 | 1.959e+01 | 3.303e+00 | 23 | 7.743e+02 |
| LSTM | 1.861e+03 | 2.850e+03 | 1.734e+02 | 5.487e+01 | 3 | 1.115e+07 |
| GNN | 1.024e+01 | 1.619e+01 | 1.354e+01 | 4.479e+00 | 41 | 6.403e+03 |
| MLP_stable | 2.805e+01 | 4.516e+01 | 3.670e+01 | 4.988e+00 | 20 | 6.500e+02 |
| LSTM_stable | 3.723e+02 | 1.065e+03 | 1.105e+02 | 2.216e+01 | 5 | 4.163e+06 |
| GNN_stable | 8.303e+00 | 1.259e+01 | 1.079e+01 | 4.071e+00 | 42 | 5.991e+03 |

### jupiter_galileans, Jupiter + 4 Galilean moons (toy circular orbits)

- N = 5, samples = 1460, dt_N = 9.419e-02
- scale: M = 1.899e+27 kg, L = 1.883e+09 m, T = 2.295e+05 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 1.227e+00 | 1.467e+00 | 6.449e+00 | 1.505e+00 | 3 | 8.097e+03 |
| LSTM | 7.089e+02 | 1.318e+03 | 1.045e+02 | 3.694e+01 | 4 | 5.751e+07 |
| GNN | 3.175e+00 | 1.222e+01 | 8.483e+00 | 2.397e+00 | 3 | 5.818e+04 |
| MLP_stable | 1.031e+00 | 1.244e+00 | 5.826e+00 | 1.393e+00 | 3 | 8.035e+03 |
| LSTM_stable | 1.254e+02 | 1.945e+02 | 6.809e+01 | 1.029e+01 | 4 | 1.897e+07 |
| GNN_stable | 3.904e+00 | 1.147e+01 | 8.392e+00 | 2.695e+00 | 3 | 1.217e+05 |

### sun_earth_only, Sun–Earth 2-body (Keplerian reference)

- N = 2, samples = 120, dt_N = 5.127e-01
- scale: M = 1.988e+30 kg, L = 1.517e+11 m, T = 5.130e+06 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 3.553e-01 | 5.220e-01 | 2.093e+00 | 6.972e-01 | 0 | 4.969e+04 |
| LSTM | 4.422e-01 | 3.947e+00 | 2.817e+00 | 7.899e-01 | 1 | 1.620e+06 |
| GNN | 2.980e-01 | 9.423e-01 | 1.976e+00 | 6.644e-01 | 0 | 1.254e+04 |
| MLP_stable | 3.552e-01 | 4.765e-01 | 2.094e+00 | 6.972e-01 | 0 | 5.103e+04 |
| LSTM_stable | 2.956e-01 | 1.204e+00 | 2.154e+00 | 6.682e-01 | 1 | 7.568e+05 |
| GNN_stable | 3.012e-01 | 8.708e-01 | 1.975e+00 | 6.743e-01 | 0 | 1.336e+04 |

### sun_planets_moon, Sun + 8 planets + Earth's Moon (10 bodies)

- N = 10, samples = 120, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 1.138e-01 | 6.459e+00 | 2.810e+00 | 3.367e-01 | 23 | 5.655e+02 |
| LSTM | 3.369e+00 | 1.620e+01 | 9.231e+00 | 1.942e+00 | 3 | 3.224e+03 |
| GNN | 5.590e-02 | 6.545e+00 | 1.293e+00 | 2.960e-01 | 42 | 3.984e+02 |
| MLP_stable | 1.144e-01 | 6.340e+00 | 2.650e+00 | 3.476e-01 | 20 | 5.200e+02 |
| LSTM_stable | 1.611e+00 | 1.377e+01 | 5.678e+00 | 1.426e+00 | 5 | 1.465e+03 |
| GNN_stable | 5.511e-02 | 6.528e+00 | 1.284e+00 | 2.943e-01 | 42 | 2.009e+02 |

### solar_system_extended, Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)

- N = 19, samples = 120, dt_N = 5.645e-04
- scale: M = 1.991e+30 kg, L = 1.423e+13 m, T = 4.658e+09 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 1.327e+00 | 1.327e+01 | 7.177e+00 | 1.173e+00 | 8 | 1.568e+02 |
| LSTM | 8.979e+00 | 2.706e+01 | 1.215e+01 | 3.954e+00 | 2 | 1.672e+03 |
| GNN | 6.625e-02 | 1.217e+01 | 1.273e+00 | 3.571e-01 | 4 | 1.546e+02 |
| MLP_stable | 1.329e+00 | 1.301e+01 | 7.149e+00 | 1.233e+00 | 8 | 1.587e+02 |
| LSTM_stable | 3.818e+00 | 2.169e+01 | 7.453e+00 | 2.651e+00 | 2 | 9.253e+02 |
| GNN_stable | 6.464e-02 | 1.220e+01 | 1.311e+00 | 3.524e-01 | 2 | 9.265e+01 |

### disc_imf_in_distribution_baseline, 25-body galaxy disc, training IMF (in-distribution sanity)

- N = 25, samples = 2500, dt_N = 2.000e-03

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 1.036e+00 | 8.434e-01 | 7.132e+00 | 1.341e+00 | 365 | 4.670e+00 |
| LSTM | 8.797e+02 | 1.309e+03 | 1.561e+02 | 3.314e+01 | 215 | 3.872e+04 |
| GNN | 1.749e+00 | 1.841e+00 | 7.926e+00 | 1.675e+00 | 482 | 5.478e+01 |
| MLP_stable | 8.116e-01 | 6.372e-01 | 5.010e+00 | 1.237e+00 | 369 | 2.571e+00 |
| LSTM_stable | 5.883e+01 | 1.811e+02 | 5.164e+01 | 7.306e+00 | 243 | 5.582e+03 |
| GNN_stable | 1.367e+00 | 1.329e+00 | 7.463e+00 | 1.471e+00 | 529 | 3.340e+01 |

## Kepler's 3rd-law check (reference integrator)

For each preset we measure the orbital period T and semi-major axis a of every non-primary body from the reference trajectory, and compare T²/a³ to the predicted 4π²/(G·M_primary). All bodies in the same preset should give the same K = T²/a³ (that's the law). The deviation is reported as a percentage. Bodies that don't complete at least one full orbit in the simulation window show NaN: increase `duration_years` to bring them in.

### inner_planets, Kepler 3rd-law check

Reference integrator (high-precision leapfrog, dt_N = 2.886e-01, 120 samples over 10 yr). For each non-primary body we report the orbital period T (median full-revolution spacing, axis-crossing detector) and the semi-major axis a = (r_min + r_max)/2, then the Kepler ratio K = T²/a³. The predicted K = 4π²/(G·M_primary) is the same for every body in this frame. Bodies whose orbit does not complete at least one full period in the simulation window show NaN, increase `duration_years` to bring them in.

| body | is_primary | mass (kg) | T (yr) | a (AU) | T²/a³ | K_pred | deviation (%) |
|---|---|---|---|---|---|---|---|
| Sun | ✓ | 1.988e+30 | — | — | — | 3.9479e+01 | — |
| Mercury |   | 3.301e+23 | 0.2442 | 0.3872 | 4.0571e+01 | 3.9479e+01 | +2.7665 |
| Venus |   | 4.868e+24 | 0.615 | 0.7233 | 3.9449e+01 | 3.9479e+01 | -0.0739 |
| Earth |   | 5.972e+24 | 1 | 1 | 3.9479e+01 | 3.9479e+01 | +0.0013 |
| Mars |   | 6.417e+23 | 1.881 | 1.524 | 3.9475e+01 | 3.9479e+01 | -0.0090 |

_The shortest-period body in this preset completes ≈ 40 full orbits in the simulation window; the longest-period body shown above completes fewer, so its T estimate is noisier._

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

### sun_earth_only, Kepler 3rd-law check

Reference integrator (high-precision leapfrog, dt_N = 5.127e-01, 120 samples over 10 yr). For each non-primary body we report the orbital period T (median full-revolution spacing, axis-crossing detector) and the semi-major axis a = (r_min + r_max)/2, then the Kepler ratio K = T²/a³. The predicted K = 4π²/(G·M_primary) is the same for every body in this frame. Bodies whose orbit does not complete at least one full period in the simulation window show NaN, increase `duration_years` to bring them in.

| body | is_primary | mass (kg) | T (yr) | a (AU) | T²/a³ | K_pred | deviation (%) |
|---|---|---|---|---|---|---|---|
| Sun | ✓ | 1.988e+30 | — | — | — | 3.9479e+01 | — |
| Earth |   | 5.972e+24 | 1 | 1 | 3.9479e+01 | 3.9479e+01 | +0.0012 |

_The shortest-period body in this preset completes ≈ 9 full orbits in the simulation window; the longest-period body shown above completes fewer, so its T estimate is noisier._

### sun_planets_moon, Kepler 3rd-law check

Reference integrator (high-precision leapfrog, dt_N = 3.208e-03, 120 samples over 10 yr). For each non-primary body we report the orbital period T (median full-revolution spacing, axis-crossing detector) and the semi-major axis a = (r_min + r_max)/2, then the Kepler ratio K = T²/a³. The predicted K = 4π²/(G·M_primary) is the same for every body in this frame. Bodies whose orbit does not complete at least one full period in the simulation window show NaN, increase `duration_years` to bring them in.

| body | is_primary | mass (kg) | T (yr) | a (AU) | T²/a³ | K_pred | deviation (%) |
|---|---|---|---|---|---|---|---|
| Sun | ✓ | 1.988e+30 | — | — | — | 3.9531e+01 | — |
| Mercury |   | 3.301e+23 | 0.2444 | 0.3872 | 4.0667e+01 | 3.9531e+01 | +2.8737 |
| Venus |   | 4.868e+24 | 0.615 | 0.7234 | 3.9499e+01 | 3.9531e+01 | -0.0818 |
| Earth |   | 5.972e+24 | 1.001 | 1 | 3.9535e+01 | 3.9531e+01 | +0.0082 |
| Mars |   | 6.417e+23 | 1.881 | 1.524 | 3.9532e+01 | 3.9531e+01 | +0.0019 |
| Jupiter |   | 1.898e+27 | - | 5.216 | - | 3.9531e+01 | - |
| Saturn |   | 5.683e+26 | - | 9.138 | - | 3.9531e+01 | - |
| Uranus |   | 8.681e+25 | - | 19.12 | - | 3.9531e+01 | - |
| Neptune |   | 1.024e+26 | - | 29.85 | - | 3.9531e+01 | - |
| Moon |   | 7.342e+22 | 0.9412 | 0.9662 | 3.8825e+01 | 3.9531e+01 | -1.7862 |

_The shortest-period body in this preset completes ≈ 40 full orbits in the simulation window; the longest-period body shown above completes fewer, so its T estimate is noisier._

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
