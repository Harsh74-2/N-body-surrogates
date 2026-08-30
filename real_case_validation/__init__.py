"""
real_case_validation
====================
Real-Case Validation pipeline for the 3D N-body surrogates.

Runs the trained MLP / LSTM / GNN checkpoints against real Solar-System
initial conditions and reports how well (or how poorly) they generalise
beyond the synthetic 25-body galaxy-disc training distribution.

See `real_case_runner.py` for the CLI entry point and `README.md` for
the user-facing overview.
"""
