"""
pipeline_config.py
==================
Single source of truth for constants and default hyperparameters shared
across the 3D N-body surrogate pipeline.

The values below are the *historical hardcoded defaults* that produced the
checkpoints and datasets already saved in the repository. They must not be
changed lightly, doing so would make old checkpoints incompatible with new
code. If you need different physics, override at the CLI or pass explicit
arguments; never edit a default here for a one-off experiment.

Coverage
--------
- Physics engine        (simulation_3d.py)
- Export / windowing    (3d_export_pipeline.py)
- PyTorch dataloader    (3d_pytorch_dataloader.py)
- MLP trainer           (mlp_train.py)
- LSTM trainer          (lstm_train.py)
- GNN trainer           (gnn_train.py)
- Evaluation            (evaluate_models.py)
- Real-case validation  (real_case_validation/*)
"""

from __future__ import annotations

import math
from enum import Enum


class ModelType(str, Enum):
    """Canonical surrogate model identifiers used across the pipeline.

    The project compares exactly these three architectures:
    MLP (per-body feed-forward), LSTM (per-body recurrent), and GNN
    (pairwise message passing).  No ``set`` / Deep-Set variant is
    implemented, so it is not listed here.
    """
    MLP = "mlp"
    LSTM = "lstm"
    GNN = "gnn"

    @classmethod
    def values(cls) -> list[str]:
        """Return the supported model-type strings in a stable order."""
        return [member.value for member in cls]


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS ENGINE DEFAULTS (simulation_3d.py)
# ══════════════════════════════════════════════════════════════════════════════
G: float = 1.0                # Gravitational constant in N-body units
SOFTENING: float = 0.1        # Plummer softening length ε
DT: float = 0.002             # Default time step (simulation time units)

DEFAULT_N_FRAMES: int = 5000  # Frames per trajectory
DEFAULT_N_SIMULATIONS: int = 5
# NOTE: there is intentionally no global DEFAULT_N_BODIES. The number of
# bodies N is an experimental variable in this project, so callers of
# `simulation_3d.py` must pass --N explicitly. Trainers infer N from the
# .npz at runtime.

# Galaxy-disc initial-condition defaults
IC_R_CORE: float = 1.0
IC_R_DISC: float = 1.5
IC_R_MAX: float = 6.0
IC_SIGMA_Z: float = 0.075
IC_M_MIN: float = 0.5         # training log-uniform mass lower bound (mass ratio ~10)
IC_M_MAX: float = 5.0         # training log-uniform mass upper bound
DISC_IMF_M_MIN: float = 0.1   # real-IMF lower bound (reference only; NOT used by the
                              # in-distribution baseline, which now matches IC_M_*)
DISC_IMF_M_MAX: float = 50.0  # real-IMF upper bound (reference only; see above)
IC_BASE_SEED: int = 42

# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT / WINDOWING DEFAULTS (3d_export_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════
WINDOW_SIZE: int = 5          # W: input window length
HORIZON: int = 1              # prediction steps ahead
STRIDE: int = 1               # stride between consecutive windows
VAL_FRAC: float = 0.1         # held-out validation fraction
TEST_FRAC: float = 0.1        # held-out test fraction (final reporting)
SPLIT_SEED: int = 42          # deterministic split seed
NORMALIZE: bool = False       # default: raw units, not z-scored
FEATURE_DIM: int = 6          # (x, y, z, vx, vy, vz)

RAW_DIR: str = "raw_data"
ML_READY_DIR: str = "ml_ready_data"
DEFAULT_NPZ: str = "ml_ready_data/dataset_3d_w5h1s1r.npz"

TRAINING_RUNS_DIR: str = "training_runs"
RESULTS_DIR: str = "results"
PLOTS_DIR: str = "plots"

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING DEFAULTS (all trainer scripts)
# ══════════════════════════════════════════════════════════════════════════════
# Generic
DEFAULT_EPOCHS: int = 20
DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-5

# Per-architecture batch sizes (memory footprint differs a lot).
# These are the values the *committed* checkpoints were trained with;
# changing them is harmless for a fresh run but would mean the
# constants diverge from the persisted weights.
MLP_BATCH_SIZE: int = 512          # mlp single-step ckpt
LSTM_BATCH_SIZE: int = 256         # lstm single-step ckpt
GNN_BATCH_SIZE: int = 128          # gnn single-step ckpt
MLP_STABLE_BATCH_SIZE: int = 256   # mlp_stable ckpt (halved for rollout K=5)
LSTM_STABLE_BATCH_SIZE: int = 256  # lstm_stable ckpt
GNN_STABLE_BATCH_SIZE: int = 128   # gnn_stable ckpt

# Default loss weights for the combined objective (see losses.py).
# w_mse   = positional MSE weight (always 1.0)
# w_energy= single-step energy drift weight
# w_rollout = K-step autoregressive rollout energy weight (0 for the
#             single-step variant, 0.1 for the stability-trained variant)
# rollout_K = number of rollout steps when w_rollout > 0
DEFAULT_W_MSE:    float = 1.0
DEFAULT_W_ENERGY: float = 0.1
DEFAULT_W_ROLLOUT_SINGLE: float = 0.0
DEFAULT_W_ROLLOUT_STABLE: float = 0.1
DEFAULT_ROLLOUT_K_SINGLE: int = 10
DEFAULT_ROLLOUT_K_STABLE: int = 5

# Per-architecture training epochs (different compute costs)
MLP_EPOCHS: int = 100
LSTM_EPOCHS: int = 80
GNN_EPOCHS: int = 50

# Per-architecture dataset sizes for simulation_3d.py
# MLP is cheapest per forward pass → more training data.
# GNN is most expensive → less training data, but still enough to learn.
MLP_NUM_SIMULATIONS: int = 20
LSTM_NUM_SIMULATIONS: int = 15
GNN_NUM_SIMULATIONS: int = 10

# MLP-specific (mlp_train.py)
MLP_HIDDEN: int = 256
MLP_LAYERS: int = 4

# LSTM-specific (lstm_train.py)
LSTM_HIDDEN: int = 256
LSTM_LAYERS: int = 2
LSTM_BIDIRECTIONAL: bool = False
LSTM_DROPOUT: float = 0.1

# GNN-specific (gnn_train.py)
GNN_HIDDEN: int = 128
GNN_LAYERS: int = 2            # number of message-passing steps
GNN_PASSING_STEPS: int = 2     # alias sometimes used in older code
GNN_DROPOUT: float = 0.0

# Optimisation / loss numerical guards
DEFAULT_EPS: float = 0.1          # softening used by loss functions
DEFAULT_GRAVITY_G: float = 1.0

# Evaluate / rollout
DEFAULT_ROLLOUT_K: int = 10
EVAL_BATCH_SIZE: int = 64
DEFAULT_DEVICE_PREFERENCE: str = "auto"   # cuda -> mps -> cpu

# Scaling sweep defaults (used by scaling_sweep.py and
# stability/eval scripts)
SWEEP_N_VALUES: list[int] = [10, 25, 50, 100]
SWEEP_ROLLOUT_K: int = 50

# ══════════════════════════════════════════════════════════════════════════════
#  REAL-CASE VALIDATION DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
# Autoregressive warm-up / rollout window used for surrogate rollouts.
REAL_CASE_WARMUP_WINDOW: int = 5

# Standard solar-system physical constants used for SI/AU rescaling.
G_SI: float = 6.67430e-11          # m³ kg⁻¹ s⁻²
AU_M: float = 1.495978707e11       # metres per astronomical unit
DAY_S: float = 86400.0             # seconds per day
SUN_MASS_KG: float = 1.98847e30
JUPITER_MASS_KG: float = 1.89813e27

# Convenience
PI: float = math.pi
TWO_PI: float = 2.0 * math.pi
