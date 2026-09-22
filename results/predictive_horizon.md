# Predictive horizon vs persistence (K=128 rollout benchmark)

k* = first rollout step at which the surrogate's cumulative rollout MSE exceeds the frozen-anchor persistence baseline (the last step at which the model is the better predictor).
`> 128` = never crosses within the benchmark horizon.

| cell | K* | ratio@32 | ratio@64 | ratio@128 |
|---|---|---|---|---|
| N10/mlp_single_step | > 128 | 0.246 | 0.418 | 0.781 |
| N10/lstm_single_step | > 128 | 0.281 | 0.377 | 0.842 |
| N10/gnn_single_step | > 128 | 0.003 | 0.004 | 0.006 |
| N10/mlp_stable | > 128 | 0.166 | 0.287 | 0.609 |
| N10/lstm_stable | > 128 | 0.263 | 0.326 | 0.530 |
| N10/gnn_stable | > 128 | 0.004 | 0.005 | 0.009 |
| N25/mlp_single_step | > 128 | 0.094 | 0.258 | 0.575 |
| N25/lstm_single_step | 121 | 0.119 | 0.346 | 1.096 |
| N25/gnn_single_step | > 128 | 0.062 | 0.057 | 0.057 |
| N25/mlp_stable | > 128 | 0.087 | 0.236 | 0.526 |
| N25/lstm_stable | 127 | 0.114 | 0.344 | 1.018 |
| N25/gnn_stable | > 128 | 0.043 | 0.042 | 0.041 |
| N50/mlp_single_step | > 128 | 0.052 | 0.180 | 0.593 |
| N50/lstm_single_step | > 128 | 0.596 | 0.595 | 0.575 |
| N50/gnn_single_step | > 128 | 0.007 | 0.007 | 0.007 |
| N50/mlp_stable | > 128 | 0.044 | 0.148 | 0.501 |
| N50/lstm_stable | > 128 | 0.582 | 0.587 | 0.577 |
| N50/gnn_stable | > 128 | 0.006 | 0.006 | 0.007 |
| N100/mlp_single_step | 127 | 0.049 | 0.184 | 1.049 |
| N100/lstm_single_step | 39 | 0.369 | 18.379 | 250.318 |
| N100/gnn_single_step | > 128 | 0.068 | 0.068 | 0.066 |
| N100/mlp_stable | > 128 | 0.040 | 0.143 | 0.594 |
| N100/lstm_stable | 45 | 0.217 | 8.656 | 144.054 |
| N100/gnn_stable | > 128 | 0.004 | 0.004 | 0.004 |
