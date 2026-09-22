# Error-percentage grid — 2026-09-22 post-retrain



Every number is read directly from the per-preset JSON artifacts of

the fresh retrain (`real_case_validation/report_N{n}/...`). Rollout

values use the calibrated post-hoc error where present. Cells above

100% of L are marked div. and excluded from the OOD mean; the disc

baseline is in-distribution and excluded from the OOD mean.

### Autoregressive rollout (mean error, % of scale length L)

| N | variant | disc (in-dist.) | full solar system | inner planets | jupiter galileans | solar extended | sun-earth only | sun-planets-moon | OOD mean |
|---|---|---|---|---|---|---|---|---|---|
| 10 | MLP | div. (172%) | div. (9486%) | 64.6% | div. (21199%) | div. (2856%) | 68.5% | div. (1044%) | 66.6% |
| 10 | MLP (stable) | div. (168%) | div. (4224%) | 63.1% | div. (5875%) | div. (2266%) | 67.0% | div. (851%) | 65.0% |
| 10 | LSTM | div. (205%) | div. (13037%) | 64.6% | div. (78358%) | div. (250628%) | 64.8% | div. (21853%) | 64.7% |
| 10 | LSTM (stable) | div. (327%) | div. (112855%) | 63.2% | div. (580789%) | div. (51369%) | 64.0% | div. (4747%) | 63.6% |
| 10 | GNN | div. (122%) | div. (8261%) | 64.6% | div. (190289%) | div. (1043%) | 74.1% | div. (191%) | 69.3% |
| 10 | GNN (stable) | div. (121%) | div. (8208%) | 64.1% | div. (172816%) | div. (1005%) | 74.0% | div. (168%) | 69.1% |
| 25 | MLP | div. (134%) | div. (2766%) | 68.1% | div. (25655%) | div. (2357%) | 69.8% | div. (376%) | 68.9% |
| 25 | MLP (stable) | div. (124%) | div. (26662%) | 67.7% | div. (16054%) | div. (3858%) | 69.7% | div. (530%) | 68.7% |
| 25 | LSTM | div. (3314%) | div. (6999525%) | 70.5% | div. (53563270%) | div. (1738106%) | 71.1% | div. (81874%) | 70.8% |
| 25 | LSTM (stable) | div. (731%) | div. (2724270%) | 74.3% | div. (29111101%) | div. (683383%) | 74.4% | div. (39683%) | 74.3% |
| 25 | GNN | div. (167%) | div. (32844%) | 63.0% | div. (73741%) | div. (2406%) | 70.3% | div. (812%) | 66.7% |
| 25 | GNN (stable) | div. (147%) | div. (15226%) | 62.6% | div. (23023%) | div. (2508%) | 70.6% | div. (729%) | 66.6% |
| 50 | MLP | div. (166%) | div. (36229%) | 64.2% | div. (221110%) | div. (835%) | 65.1% | div. (191%) | 64.6% |
| 50 | MLP (stable) | div. (147%) | div. (34701%) | 64.1% | div. (74252%) | div. (1015%) | 65.1% | div. (218%) | 64.6% |
| 50 | LSTM | div. (130%) | div. (10654%) | 63.1% | div. (62722%) | div. (2063%) | 65.0% | div. (453%) | 64.0% |
| 50 | LSTM (stable) | div. (127%) | div. (27494%) | 64.2% | div. (141242%) | div. (20930%) | 66.1% | div. (302%) | 65.2% |
| 50 | GNN | div. (177%) | div. (20491%) | 66.6% | div. (90125%) | div. (4389%) | 79.4% | div. (970%) | 73.0% |
| 50 | GNN (stable) | div. (162%) | div. (23918%) | 66.4% | div. (96827%) | div. (4297%) | 79.8% | div. (954%) | 73.1% |
| 100 | MLP | div. (288%) | div. (33109%) | 63.0% | div. (134417%) | div. (3172%) | 68.3% | div. (806%) | 65.6% |
| 100 | MLP (stable) | div. (185%) | div. (20775%) | 63.7% | div. (38465%) | div. (2361%) | 69.4% | div. (551%) | 66.6% |
| 100 | LSTM | div. (315%) | div. (146396%) | div. (137%) | div. (1063174%) | div. (22341%) | div. (137%) | div. (5416%) | &mdash; |
| 100 | LSTM (stable) | div. (301%) | div. (39685%) | 97.9% | div. (195847%) | div. (10685%) | div. (114%) | div. (2174%) | 97.9% |
| 100 | GNN | div. (115%) | div. (22643%) | 64.2% | div. (186327%) | div. (5185%) | 80.1% | div. (1410%) | 72.2% |
| 100 | GNN (stable) | div. (180%) | div. (29352%) | 73.0% | div. (113421%) | div. (6390%) | 92.0% | div. (3432%) | 82.5% |

### Single-step (mean error, % of scale length L)

| N | variant | disc (in-dist.) | full solar system | inner planets | jupiter galileans | solar extended | sun-earth only | sun-planets-moon | OOD mean |
|---|---|---|---|---|---|---|---|---|---|
| 10 | MLP | 0.0% | 0.8% | 29.4% | 11.1% | 0.5% | 25.3% | 0.8% | 11.3% |
| 10 | MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 10 | LSTM | 0.0% | 2.9% | 29.1% | 10.7% | 11.4% | 24.8% | 3.3% | 13.7% |
| 10 | LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 10 | GNN | 0.0% | 0.9% | 29.7% | 11.3% | 0.3% | 25.5% | 1.0% | 11.4% |
| 10 | GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 25 | MLP | 0.0% | 1.0% | 29.5% | 11.1% | 2.0% | 25.3% | 1.1% | 11.7% |
| 25 | MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 25 | LSTM | 0.0% | 3.3% | 29.4% | 10.8% | 9.7% | 24.7% | 3.8% | 13.6% |
| 25 | LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 25 | GNN | 0.0% | 1.1% | 29.8% | 11.4% | 0.6% | 25.6% | 1.2% | 11.6% |
| 25 | GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 50 | MLP | 0.0% | 1.1% | 29.6% | 11.2% | 1.3% | 25.4% | 1.1% | 11.6% |
| 50 | MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 50 | LSTM | 0.0% | 1.4% | 29.8% | 11.3% | 1.4% | 25.5% | 1.6% | 11.8% |
| 50 | LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 50 | GNN | 0.0% | 1.0% | 29.9% | 11.4% | 0.4% | 25.8% | 1.1% | 11.6% |
| 50 | GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 100 | MLP | 0.0% | 1.1% | 29.4% | 11.3% | 1.1% | 25.5% | 1.2% | 11.6% |
| 100 | MLP (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 100 | LSTM | 0.0% | 1.0% | 29.6% | 11.1% | 1.2% | 25.4% | 1.1% | 11.6% |
| 100 | LSTM (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| 100 | GNN | 0.0% | 1.1% | 29.8% | 11.4% | 0.7% | 25.6% | 1.2% | 11.6% |
| 100 | GNN (stable) | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
