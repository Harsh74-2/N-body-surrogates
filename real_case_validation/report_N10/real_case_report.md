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
| MLP | 2.197e-01 | 8.117e-01 | 2.028e+00 | 6.223e-01 | 1 | 2.383e+05 |
| LSTM | 2.321e-01 | 1.235e+00 | 2.026e+00 | 6.674e-01 | 1 | 5.281e+05 |
| GNN | 2.212e-01 | 1.639e+00 | 2.046e+00 | 6.172e-01 | 1 | 1.849e+03 |
| MLP_stable | 2.177e-01 | 7.359e-01 | 2.030e+00 | 6.195e-01 | 1 | 1.755e+05 |
| LSTM_stable | 2.122e-01 | 9.025e-01 | 2.022e+00 | 6.340e-01 | 1 | 3.775e+05 |
| GNN_stable | 2.193e-01 | 1.466e+00 | 2.045e+00 | 6.141e-01 | 1 | 1.734e+03 |

### full_solar_system, All 8 planets + Sun

- N = 9, samples = 2400, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 2.882e+00 | 6.407e+00 | 7.262e+00 | 2.419e+00 | 54 | 5.636e+04 |
| LSTM | 4.752e+00 | 8.986e+00 | 1.138e+01 | 3.147e+00 | 8 | 4.716e+03 |
| GNN | 8.245e+00 | 6.354e+02 | 1.123e+01 | 4.129e+00 | 90 | 1.730e+06 |
| MLP_stable | 3.489e+00 | 8.705e+00 | 1.222e+01 | 2.266e+00 | 56 | 1.172e+03 |
| LSTM_stable | 5.150e+01 | 3.368e+01 | 3.931e+01 | 8.287e+00 | 9 | 5.645e+03 |
| GNN_stable | 8.533e+00 | 5.246e+02 | 1.151e+01 | 4.214e+00 | 90 | 1.171e+06 |

### jupiter_galileans, Jupiter + 4 Galilean moons (toy circular orbits)

- N = 5, samples = 1460, dt_N = 9.419e-02
- scale: M = 1.899e+27 kg, L = 1.883e+09 m, T = 2.295e+05 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 7.942e-01 | 1.169e+00 | 4.509e+00 | 1.252e+00 | 3 | 2.714e+04 |
| LSTM | 2.183e+00 | 4.496e+00 | 6.411e+00 | 2.161e+00 | 2 | 5.204e+04 |
| GNN | 2.567e+00 | 3.425e+02 | 6.831e+00 | 2.322e+00 | 3 | 3.591e+06 |
| MLP_stable | 7.997e-01 | 1.008e+00 | 4.055e+00 | 1.254e+00 | 3 | 7.981e+03 |
| LSTM_stable | 7.661e+00 | 6.417e+00 | 1.709e+01 | 3.351e+00 | 2 | 5.745e+04 |
| GNN_stable | 2.660e+00 | 2.718e+02 | 6.852e+00 | 2.354e+00 | 3 | 3.126e+06 |

### sun_earth_only, Sun–Earth 2-body (Keplerian reference)

- N = 2, samples = 120, dt_N = 5.127e-01
- scale: M = 1.988e+30 kg, L = 1.517e+11 m, T = 5.130e+06 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 2.945e-01 | 6.206e-01 | 1.959e+00 | 6.385e-01 | 0 | 4.307e+05 |
| LSTM | 2.868e-01 | 5.940e-01 | 1.942e+00 | 6.158e-01 | 1 | 1.573e+06 |
| GNN | 2.967e-01 | 2.281e+00 | 1.975e+00 | 6.545e-01 | 0 | 8.022e+03 |
| MLP_stable | 2.995e-01 | 5.324e-01 | 1.965e+00 | 6.359e-01 | 0 | 3.586e+05 |
| LSTM_stable | 2.969e-01 | 4.847e-01 | 1.931e+00 | 6.225e-01 | 1 | 1.117e+06 |
| GNN_stable | 2.958e-01 | 2.087e+00 | 1.975e+00 | 6.535e-01 | 0 | 7.996e+03 |

### sun_planets_moon, Sun + 8 planets + Earth's Moon (10 bodies)

- N = 10, samples = 120, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 4.366e-02 | 5.708e+00 | 1.096e+00 | 2.708e-01 | 54 | 6.683e+02 |
| LSTM | 1.975e+00 | 7.984e+00 | 1.114e+01 | 1.301e+00 | 8 | 8.538e+02 |
| GNN | 2.527e-02 | 1.090e+01 | 6.992e-01 | 2.132e-01 | 90 | 3.862e+02 |
| MLP_stable | 4.177e-02 | 5.895e+00 | 1.026e+00 | 2.667e-01 | 56 | 4.299e+02 |
| LSTM_stable | 6.795e-01 | 6.924e+00 | 6.459e+00 | 7.693e-01 | 9 | 5.940e+02 |
| GNN_stable | 2.438e-02 | 1.043e+01 | 6.917e-01 | 2.094e-01 | 90 | 2.072e+02 |

### solar_system_extended, Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)

- N = 19, samples = 120, dt_N = 5.645e-04
- scale: M = 1.991e+30 kg, L = 1.423e+13 m, T = 4.658e+09 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 4.445e-02 | 1.138e+01 | 1.141e+00 | 2.872e-01 | 50 | 1.870e+02 |
| LSTM | 7.635e+00 | 2.183e+01 | 1.755e+01 | 3.390e+00 | 1 | 2.836e+02 |
| GNN | 2.101e-02 | 1.978e+01 | 5.993e-01 | 1.975e-01 | 96 | 1.536e+02 |
| MLP_stable | 3.585e-02 | 1.143e+01 | 1.039e+00 | 2.451e-01 | 56 | 1.325e+02 |
| LSTM_stable | 3.103e+00 | 1.715e+01 | 1.108e+01 | 2.161e+00 | 1 | 1.999e+02 |
| GNN_stable | 2.129e-02 | 1.904e+01 | 6.034e-01 | 1.994e-01 | 96 | 1.064e+02 |

### disc_imf_in_distribution_baseline, 25-body galaxy disc, training IMF (in-distribution sanity)

- N = 25, samples = 2500, dt_N = 2.000e-03

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 1.736e+00 | 1.461e+00 | 7.917e+00 | 1.719e+00 | 254 | 1.111e+01 |
| LSTM | 2.409e+00 | 2.848e+00 | 9.607e+00 | 2.049e+00 | 270 | 3.386e+01 |
| GNN | 1.006e+00 | 3.739e+00 | 5.738e+00 | 1.223e+00 | 435 | 1.273e+02 |
| MLP_stable | 1.889e+00 | 1.517e+00 | 1.170e+01 | 1.675e+00 | 266 | 1.760e+01 |
| LSTM_stable | 1.062e+01 | 7.100e+00 | 3.650e+01 | 3.272e+00 | 290 | 4.048e+01 |
| GNN_stable | 9.751e-01 | 3.820e+00 | 6.408e+00 | 1.206e+00 | 457 | 8.856e+01 |

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
