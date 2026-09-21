# Cross-N audit (single-step)



Post-audit retrain (2026-09).

Single-step test metrics (held-out windows, in-distribution):
mean squared error of the next-frame prediction, the
energy-relative drift of that prediction, and the per-step
latency. Source: `results/all_eval.json`.

| N | variant | params | single-step MSE | energy drift | latency (ms) |
|---|---|---|---|---|---|
| 10 | MLP | 210,182 | 1.17e-05 | 1.20e-06 | 0.249 |
| 10 | MLP (stable) | 210,182 | 1.17e-05 | 1.29e-06 | 0.250 |
| 10 | LSTM | 865,542 | 7.69e-06 | 2.06e-06 | 0.383 |
| 10 | LSTM (stable) | 865,542 | 7.69e-06 | 2.05e-06 | 0.379 |
| 10 | GNN | 168,199 | 2.87e-06 | 5.56e-07 | 1.883 |
| 10 | GNN (stable) | 168,199 | 2.87e-06 | 4.65e-07 | 1.880 |
| 25 | MLP | 210,182 | 4.96e-06 | 3.02e-07 | 0.248 |
| 25 | MLP (stable) | 210,182 | 4.97e-06 | 2.74e-07 | 0.244 |
| 25 | LSTM | 865,542 | 5.87e-06 | 2.14e-07 | 0.445 |
| 25 | LSTM (stable) | 865,542 | 5.87e-06 | 2.08e-07 | 0.445 |
| 25 | GNN | 168,199 | 4.94e-06 | 2.28e-07 | 3.888 |
| 25 | GNN (stable) | 168,199 | 4.94e-06 | 4.59e-07 | 3.877 |
| 50 | MLP | 210,182 | 4.47e-06 | 1.91e-07 | 0.254 |
| 50 | MLP (stable) | 210,182 | 4.47e-06 | 1.65e-07 | 0.251 |
| 50 | LSTM | 865,542 | 3.28e-06 | 1.49e-07 | 0.776 |
| 50 | LSTM (stable) | 865,542 | 3.28e-06 | 1.48e-07 | 0.780 |
| 50 | GNN | 168,199 | 1.75e-06 | 1.96e-07 | 15.802 |
| 50 | GNN (stable) | 168,199 | 1.75e-06 | 1.96e-07 | 15.928 |
| 100 | MLP | 210,182 | 3.23e-06 | 1.63e-07 | 0.318 |
| 100 | MLP (stable) | 210,182 | 3.23e-06 | 1.43e-07 | 0.316 |
| 100 | LSTM | 865,542 | 2.53e-06 | 1.08e-07 | 1.494 |
| 100 | LSTM (stable) | 865,542 | 2.53e-06 | 1.06e-07 | 1.480 |
| 100 | GNN | 168,199 | 2.11e-06 | 1.76e-07 | 64.930 |
| 100 | GNN (stable) | 168,199 | 2.11e-06 | 1.91e-07 | 65.611 |
