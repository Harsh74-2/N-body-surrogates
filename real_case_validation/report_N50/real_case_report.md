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
| MLP | 2.534e-01 | 6.165e-01 | 2.160e+00 | 6.538e-01 | 1 | 1.076e+05 |
| LSTM | 2.401e-01 | 6.819e-01 | 2.048e+00 | 6.383e-01 | 1 | 7.318e+03 |
| GNN | 2.410e-01 | 1.048e+00 | 2.051e+00 | 6.758e-01 | 1 | 1.100e+04 |
| MLP_stable | 2.505e-01 | 5.970e-01 | 2.147e+00 | 6.502e-01 | 1 | 9.644e+04 |
| LSTM_stable | 2.403e-01 | 7.380e-01 | 2.049e+00 | 6.405e-01 | 1 | 1.015e+04 |
| GNN_stable | 2.408e-01 | 1.044e+00 | 2.051e+00 | 6.755e-01 | 1 | 1.110e+04 |

### full_solar_system, All 8 planets + Sun

- N = 9, samples = 2400, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 5.313e+01 | 3.574e+02 | 4.937e+01 | 6.466e+00 | 51 | 4.683e+02 |
| LSTM | 1.531e+01 | 4.662e+01 | 1.769e+01 | 4.985e+00 | 70 | 2.429e+02 |
| GNN | 1.241e+01 | 1.638e+01 | 1.551e+01 | 4.926e+00 | 77 | 3.585e+04 |
| MLP_stable | 4.790e+01 | 2.335e+02 | 4.693e+01 | 6.201e+00 | 40 | 5.633e+03 |
| LSTM_stable | 2.654e+01 | 3.491e+01 | 2.168e+01 | 6.659e+00 | 30 | 7.133e+02 |
| GNN_stable | 1.316e+01 | 1.622e+01 | 1.586e+01 | 5.150e+00 | 77 | 3.587e+04 |

### jupiter_galileans, Jupiter + 4 Galilean moons (toy circular orbits)

- N = 5, samples = 1460, dt_N = 9.419e-02
- scale: M = 1.899e+27 kg, L = 1.883e+09 m, T = 2.295e+05 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 8.252e+00 | 4.177e+01 | 2.063e+01 | 2.996e+00 | 3 | 6.646e+03 |
| LSTM | 1.397e+00 | 1.583e+00 | 7.256e+00 | 1.680e+00 | 3 | 3.014e+03 |
| GNN | 3.971e+00 | 8.924e+00 | 7.783e+00 | 2.896e+00 | 3 | 1.877e+05 |
| MLP_stable | 1.268e+00 | 4.168e+00 | 6.230e+00 | 1.592e+00 | 3 | 1.294e+04 |
| LSTM_stable | 3.917e+00 | 6.412e+00 | 9.910e+00 | 2.388e+00 | 3 | 1.386e+03 |
| GNN_stable | 4.369e+00 | 1.128e+01 | 8.277e+00 | 3.027e+00 | 3 | 2.032e+05 |

### sun_earth_only, Sun–Earth 2-body (Keplerian reference)

- N = 2, samples = 120, dt_N = 5.127e-01
- scale: M = 1.988e+30 kg, L = 1.517e+11 m, T = 5.130e+06 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 3.494e-01 | 4.563e-01 | 2.096e+00 | 6.791e-01 | 0 | 1.554e+05 |
| LSTM | 3.313e-01 | 3.453e-01 | 1.980e+00 | 6.568e-01 | 0 | 1.071e+04 |
| GNN | 4.055e-01 | 8.356e-01 | 2.228e+00 | 8.375e-01 | 0 | 1.232e+04 |
| MLP_stable | 3.471e-01 | 4.344e-01 | 2.084e+00 | 6.797e-01 | 0 | 1.328e+05 |
| LSTM_stable | 3.295e-01 | 3.561e-01 | 1.993e+00 | 6.525e-01 | 0 | 2.016e+04 |
| GNN_stable | 4.055e-01 | 8.055e-01 | 2.223e+00 | 8.405e-01 | 0 | 1.268e+04 |

### sun_planets_moon, Sun + 8 planets + Earth's Moon (10 bodies)

- N = 10, samples = 120, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 3.009e-02 | 5.976e+00 | 1.103e+00 | 2.205e-01 | 51 | 4.425e+02 |
| LSTM | 4.006e-02 | 7.143e+00 | 8.505e-01 | 2.658e-01 | 70 | 3.469e+01 |
| GNN | 3.683e-02 | 7.687e+00 | 7.931e-01 | 2.641e-01 | 75 | 8.784e+02 |
| MLP_stable | 4.017e-02 | 5.942e+00 | 1.449e+00 | 2.366e-01 | 40 | 4.494e+02 |
| LSTM_stable | 6.720e-02 | 7.117e+00 | 1.768e+00 | 3.047e-01 | 30 | 3.315e+01 |
| GNN_stable | 3.688e-02 | 7.653e+00 | 7.904e-01 | 2.636e-01 | 74 | 8.135e+02 |

### solar_system_extended, Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)

- N = 19, samples = 120, dt_N = 5.645e-04
- scale: M = 1.991e+30 kg, L = 1.423e+13 m, T = 4.658e+09 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 2.031e-01 | 1.153e+01 | 2.674e+00 | 5.305e-01 | 21 | 1.275e+02 |
| LSTM | 4.639e-02 | 1.301e+01 | 1.087e+00 | 2.873e-01 | 32 | 1.322e+01 |
| GNN | 2.872e-02 | 1.410e+01 | 8.270e-01 | 2.314e-01 | 71 | 2.728e+02 |
| MLP_stable | 2.169e-01 | 1.162e+01 | 2.664e+00 | 5.517e-01 | 19 | 1.174e+02 |
| LSTM_stable | 6.599e-01 | 1.415e+01 | 6.246e+00 | 9.276e-01 | 10 | 1.163e+01 |
| GNN_stable | 2.997e-02 | 1.399e+01 | 8.512e-01 | 2.354e-01 | 70 | 2.505e+02 |

### disc_imf_in_distribution_baseline, 25-body galaxy disc, training IMF (in-distribution sanity)

- N = 25, samples = 2500, dt_N = 2.000e-03

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 2.090e+00 | 5.331e+00 | 1.770e+01 | 1.664e+00 | 251 | 2.143e+02 |
| LSTM | 8.335e-01 | 6.154e-01 | 3.747e+00 | 1.302e+00 | 366 | 8.096e-01 |
| GNN | 1.804e+00 | 1.318e+00 | 7.746e+00 | 1.769e+00 | 443 | 2.708e+00 |
| MLP_stable | 1.097e+00 | 8.142e-01 | 5.343e+00 | 1.467e+00 | 249 | 1.985e+00 |
| LSTM_stable | 7.766e-01 | 5.834e-01 | 3.419e+00 | 1.266e+00 | 343 | 1.371e+00 |
| GNN_stable | 1.601e+00 | 1.170e+00 | 7.638e+00 | 1.618e+00 | 463 | 2.675e+00 |

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
