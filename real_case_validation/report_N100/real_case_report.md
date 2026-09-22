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
| MLP | 2.392e-01 | 1.233e+00 | 2.062e+00 | 6.560e-01 | 1 | 5.438e+04 |
| LSTM | 7.805e-01 | 1.939e+02 | 3.609e+00 | 1.302e+00 | 1 | 7.291e+07 |
| GNN | 2.317e-01 | 6.753e-01 | 2.048e+00 | 6.253e-01 | 1 | 4.821e+03 |
| MLP_stable | 2.519e-01 | 8.049e-01 | 2.133e+00 | 6.692e-01 | 1 | 5.042e+04 |
| LSTM_stable | 5.398e-01 | 1.735e+02 | 3.109e+00 | 1.068e+00 | 1 | 8.939e+07 |
| GNN_stable | 2.375e-01 | 7.262e-01 | 2.042e+00 | 6.910e-01 | 1 | 1.319e+04 |

### full_solar_system, All 8 planets + Sun

- N = 9, samples = 2400, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 2.695e+01 | 1.021e+02 | 2.770e+01 | 7.042e+00 | 33 | 4.180e+03 |
| LSTM | 6.459e+01 | 4.437e+04 | 3.313e+01 | 1.075e+01 | 22 | 1.218e+08 |
| GNN | 1.405e+01 | 3.002e+01 | 1.472e+01 | 5.303e+00 | 40 | 1.075e+05 |
| MLP_stable | 1.889e+01 | 3.614e+01 | 2.201e+01 | 5.744e+00 | 30 | 2.382e+05 |
| LSTM_stable | 2.053e+01 | 4.454e+04 | 2.858e+01 | 6.049e+00 | 26 | 1.589e+08 |
| GNN_stable | 1.412e+01 | 1.414e+01 | 1.608e+01 | 5.168e+00 | 53 | 7.915e+03 |

### jupiter_galileans, Jupiter + 4 Galilean moons (toy circular orbits)

- N = 5, samples = 1460, dt_N = 9.419e-02
- scale: M = 1.899e+27 kg, L = 1.883e+09 m, T = 2.295e+05 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 6.391e+00 | 2.140e+01 | 1.137e+01 | 3.154e+00 | 3 | 2.714e+04 |
| LSTM | 8.599e+01 | 2.286e+04 | 4.230e+01 | 1.355e+01 | 3 | 5.260e+08 |
| GNN | 3.235e+00 | 7.257e+00 | 7.076e+00 | 2.597e+00 | 3 | 1.270e+05 |
| MLP_stable | 8.978e-01 | 1.587e+00 | 5.255e+00 | 1.341e+00 | 3 | 2.756e+05 |
| LSTM_stable | 1.203e+01 | 2.592e+04 | 1.490e+01 | 5.332e+00 | 3 | 6.628e+08 |
| GNN_stable | 2.841e+00 | 5.954e+00 | 6.791e+00 | 2.386e+00 | 3 | 7.495e+04 |

### sun_earth_only, Sun–Earth 2-body (Keplerian reference)

- N = 2, samples = 120, dt_N = 5.127e-01
- scale: M = 1.988e+30 kg, L = 1.517e+11 m, T = 5.130e+06 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 3.139e-01 | 7.645e-01 | 1.963e+00 | 6.684e-01 | 0 | 6.144e+04 |
| LSTM | 8.510e-01 | 1.749e+02 | 3.707e+00 | 1.276e+00 | 0 | 1.017e+08 |
| GNN | 2.868e-01 | 5.397e-01 | 1.972e+00 | 6.730e-01 | 0 | 1.942e+04 |
| MLP_stable | 3.395e-01 | 5.203e-01 | 2.003e+00 | 6.960e-01 | 0 | 5.643e+04 |
| LSTM_stable | 6.495e-01 | 1.602e+02 | 3.192e+00 | 1.068e+00 | 0 | 1.362e+08 |
| GNN_stable | 3.364e-01 | 5.936e-01 | 1.977e+00 | 7.754e-01 | 0 | 8.627e+03 |

### sun_planets_moon, Sun + 8 planets + Earth's Moon (10 bodies)

- N = 10, samples = 120, dt_N = 3.208e-03
- scale: M = 1.991e+30 kg, L = 4.470e+12 m, T = 8.198e+08 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 7.555e-02 | 8.270e+00 | 1.679e+00 | 3.428e-01 | 33 | 7.162e+02 |
| LSTM | 3.201e-01 | 1.495e+02 | 2.292e+00 | 7.118e-01 | 22 | 5.494e+05 |
| GNN | 6.411e-02 | 7.082e+00 | 1.212e+00 | 3.231e-01 | 40 | 4.151e+02 |
| MLP_stable | 6.956e-02 | 7.652e+00 | 1.764e+00 | 3.256e-01 | 30 | 6.828e+02 |
| LSTM_stable | 1.846e-01 | 1.373e+02 | 1.725e+00 | 5.521e-01 | 26 | 5.235e+05 |
| GNN_stable | 7.102e-02 | 6.860e+00 | 1.153e+00 | 3.654e-01 | 51 | 1.619e+02 |

### solar_system_extended, Sun + 8 planets + Moon + 5 dwarfs + 4 Galilean moons (19 bodies)

- N = 19, samples = 120, dt_N = 5.645e-04
- scale: M = 1.991e+30 kg, L = 1.423e+13 m, T = 4.658e+09 s

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 1.505e-01 | 1.546e+01 | 1.998e+00 | 5.307e-01 | 28 | 2.047e+02 |
| LSTM | 5.210e-01 | 2.062e+02 | 3.079e+00 | 9.860e-01 | 16 | 1.151e+05 |
| GNN | 7.995e-02 | 1.290e+01 | 1.483e+00 | 3.662e-01 | 26 | 1.843e+02 |
| MLP_stable | 1.441e-01 | 1.466e+01 | 2.481e+00 | 4.992e-01 | 23 | 1.944e+02 |
| LSTM_stable | 3.376e-01 | 1.867e+02 | 2.448e+00 | 8.045e-01 | 21 | 1.336e+05 |
| GNN_stable | 4.040e-02 | 1.394e+01 | 9.528e-01 | 2.811e-01 | 57 | 3.792e+01 |

### disc_imf_in_distribution_baseline, 25-body galaxy disc, training IMF (in-distribution sanity)

- N = 25, samples = 2500, dt_N = 2.000e-03

| model | MSE (pos) | MSE (state) | max err / L | mean err / L | frames to ½L error | max energy drift |
|---|---|---|---|---|---|---|
| MLP | 6.173e+00 | 3.404e+01 | 2.151e+01 | 2.878e+00 | 267 | 1.207e+03 |
| LSTM | 9.830e+00 | 7.304e+03 | 3.700e+01 | 3.153e+00 | 118 | 3.230e+05 |
| GNN | 7.602e-01 | 8.768e-01 | 5.175e+00 | 1.148e+00 | 493 | 2.301e+01 |
| MLP_stable | 2.421e+00 | 4.645e+00 | 1.528e+01 | 1.853e+00 | 289 | 1.746e+02 |
| LSTM_stable | 5.103e+00 | 1.103e+04 | 1.313e+01 | 3.008e+00 | 125 | 4.262e+05 |
| GNN_stable | 2.062e+00 | 1.612e+00 | 1.052e+01 | 1.795e+00 | 307 | 4.948e+00 |

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
