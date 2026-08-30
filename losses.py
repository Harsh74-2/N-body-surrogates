"""
losses.py
=========
Loss functions for training neural surrogates on 3D N-body trajectories.

Three losses, used together to give the model every signal your project
benchmarks:

1. `mse_loss(pred, target)`
       Per-feature mean-squared error between the predicted next state
       and the ground-truth next state. The single-step positional
       accuracy metric (project metric #1).

2. `energy_drift_loss(state_pred, state_true, mass, eps, g)`
       |E(state_pred) − E(state_true)| / |E(state_true)|, where E is
       the total mechanical energy (KE + softened PE). Direct penalty
       for energy non-conservation on a single step.

3. `rollout_energy_loss(model, window_init, mass, ref_state, eps, g, K)`
       Autoregressive K-step rollout seeded with the true W-window the
       model was trained on. At each step the model predicts the next
       state from the current W-window, the window is shifted with the
       prediction, and the energy drift relative to `ref_state` is
       accumulated. Gradients flow through every rollout step (the model
       is in train() mode and we don't detach), so the model is trained
       to stay close to energy conservation even when its own
       predictions are the input -- the standard fix for autoregressive
       drift in neural surrogates (Greydanus et al. 2019;
       Sanchez-Gonzalez et al. 2020).

       The rollout uses the model's full W-window forward (with message
       passing for the GNN), matching how every model is evaluated in
       `evaluate_models.py` and `stability_benchmark.py`. An earlier
       version called `model.step` (a degenerate W=1 window) here; for
       the GNN that skipped all message passing and trained the
       stability term on an out-of-distribution path, which is why the
       stability-trained GNN's energy drift got *worse*. Each rollout
       step's forward is gradient-checkpointed so K-step BPTT fits on a
       48 GB GPU at the training batch size.

Conventions
-----------
- States are tensors of shape (..., N, F) where F = 6:
        [0:3] = position (x, y, z)
        [3:6] = velocity (vx, vy, vz)
- Mass is shape (..., N), one scalar per body, normalised so the sum
  over N is 1 in your generated data. We do *not* re-normalise here.
- Softening eps and gravitational constant g are configurable. Default
  values match `simulation_3d.py`: eps = 0.1, g = 1.0.

Notes on numerical stability
----------------------------
Energy drift ratios can blow up when the reference energy is near
zero (rare in practice for a bound system, but possible during early
training). We guard with a small epsilon floor on |E| in the
denominator, so the loss stays finite even if the initial state is
near zero energy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pipeline_config import DEFAULT_EPS, DEFAULT_GRAVITY_G


# ── Energy helpers ───────────────────────────────────────────────────────────
def kinetic_energy(vel: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """
    Total kinetic energy: KE = ½ Σᵢ mᵢ |vᵢ|².

    vel  : (..., N, 3)
    mass : (..., N)
    Returns: (...,), one scalar per leading batch dim
    """
    v_sq = (vel ** 2).sum(dim=-1)              # (..., N)
    return 0.5 * (mass * v_sq).sum(dim=-1)     # (...,)


def potential_energy(pos: torch.Tensor,
                     mass: torch.Tensor,
                     eps: float) -> torch.Tensor:
    """
    Softened gravitational PE: PE = −Σ_{i<j} mᵢ mⱼ / √(|rᵢ−rⱼ|² + ε²).

    Vectorised via broadcasting, same O(N²/2) pattern as
    `simulation_3d.total_energy`. No Python loop.

    pos  : (..., N, 3)
    mass : (..., N)
    Returns: (...,), one scalar per leading batch dim
    """
    # Pairwise displacements: r_j - r_i   shape (..., N, N, 3)
    diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)
    # Softened distance
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps * eps)        # (..., N, N)
    # Mass product m_i * m_j   shape (..., N, N)
    m_prod = mass.unsqueeze(-1) * mass.unsqueeze(-2)
    # Sum over upper triangle only (i<j), then multiply by 2 (symmetry).
    # We achieve the upper-triangular sum via the identity:
    #   Σ_{i<j} a_ij = ½ (Σ a_ij − diagonal(a_ij))
    # Diagonal of dist is sqrt(0 + eps^2) = eps; diagonal of m_prod is
    # m_i^2. We subtract those before the half-multiplier to keep PE
    # finite and well-defined.
    inv_r    = m_prod / dist                                    # (..., N, N)
    diag     = torch.diagonal(inv_r, dim1=-2, dim2=-1)          # (..., N)
    sum_off  = inv_r.sum(dim=(-1, -2)) - diag.sum(dim=-1)       # (...,)
    return -0.5 * sum_off


def total_energy(pos: torch.Tensor,
                 vel: torch.Tensor,
                 mass: torch.Tensor,
                 eps: float = DEFAULT_EPS,
                 g: float = DEFAULT_GRAVITY_G) -> torch.Tensor:
    """
    Total mechanical energy E = KE + PE.

    Newtonian convention, KE has no g; g only appears in PE:

        KE = ½ Σᵢ mᵢ |vᵢ|²
        PE = − g · Σ_{i<j} mᵢ mⱼ / √(|rᵢ−rⱼ|² + ε²)

    This matches `simulation_3d.total_energy` so the surrogate is
    trained against the same formula the engine and validation suite
    use. (Earlier versions of this file multiplied KE by g as well -
    that was wrong: it left |ΔE/E₀| invariant under common scaling,
    so energy *conservation* tests passed, but the virial ratio and
    any loss that compared energies against absolute references
    were distorted.)

    pos  : (..., N, 3)
    vel  : (..., N, 3)
    mass : (..., N)
    Returns: (...,)
    """
    pe = potential_energy(pos, mass, eps) * g
    ke = kinetic_energy(vel, mass)         # NO g, Newtonian convention
    return ke + pe


# ── Loss 1: positional MSE ───────────────────────────────────────────────────
def mse_loss(pred: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
    """
    Per-feature mean-squared error, summed over the last dim and
    averaged over the rest. Same as `torch.nn.functional.mse_loss`
    with `reduction='mean'`, exposed here so all three losses live
    in one module.

    pred, target : (..., N, F): typically (B, N, F)
    """
    return F.mse_loss(pred, target)


# ── Loss 2: single-step energy drift ─────────────────────────────────────────
def energy_drift_loss(state_pred: torch.Tensor,
                      state_true: torch.Tensor,
                      mass: torch.Tensor,
                      eps: float = DEFAULT_EPS,
                      g: float = DEFAULT_GRAVITY_G,
                      eps_floor: float = 1e-8) -> torch.Tensor:
    """
    |E(state_pred) − E(state_true)| / max(|E(state_true)|, eps_floor).

    A direct penalty for energy non-conservation on a single step.
    Averaged across the batch.

    state_pred, state_true : (B, N, 6)
    mass                   : (B, N)
    """
    e_pred = total_energy(state_pred[..., :3], state_pred[..., 3:6],
                          mass, eps=eps, g=g)
    e_true = total_energy(state_true[..., :3], state_true[..., 3:6],
                          mass, eps=eps, g=g)
    denom  = torch.clamp(e_true.abs(), min=eps_floor)
    return ((e_pred - e_true).abs() / denom).mean()


# ── Loss 3: K-step rollout energy drift ──────────────────────────────────────
def rollout_energy_loss(model: torch.nn.Module,
                        window_init: torch.Tensor,
                        mass: torch.Tensor,
                        ref_state: torch.Tensor | None = None,
                        eps: float = DEFAULT_EPS,
                        g: float = DEFAULT_GRAVITY_G,
                        K: int = 10,
                        eps_floor: float = 1e-8) -> torch.Tensor:
    """
    Autoregressive K-step rollout seeded with a true W-window.

    At each step k = 1..K:
        y_pred   = model(window)          # full W-window forward (with
                                          # message passing for the GNN)
        window   = shift(window, y_pred)  # slide the window with the
                                          # prediction (autoregressive)

    The loss is the mean of |E(y_pred_k) - E(ref_state)| /
    max(|E(ref_state)|, eps_floor) over k = 1..K, averaged across the
    batch. `ref_state` is the true next state (the frame the first
    prediction targets); if it is not given the last frame of the seed
    window is used.

    Gradients flow through every rollout step, the model is in train()
    mode and we deliberately do NOT detach. This is the standard recipe
    for training autoregressive-stable neural surrogates (Greydanus et
    al. 2019; Sanchez-Gonzalez et al. 2020). The rollout is identical
    to the sliding-window rollout used for evaluation in
    `evaluate_models.py` and `stability_benchmark.py`, so the stability
    term optimises the same path that is later measured.

    Each rollout step's forward is wrapped in `torch.utils.checkpoint`
    *for message-passing models* (the GNN) so that K-step backprop-
    through-time retains only the step inputs and recomputes activations
    during backward. Without this the GNN's (B, N, N, hidden) message
    tensors would make K=5 BPTT peak well above 48 GB at N >= 50; with
    checkpointing the peak is roughly one forward's worth of activations.
    The MLP and LSTM hold no N^2 tensor (their K=5 BPTT fits easily), so
    they run the rollout directly -- this also avoids recomputing the
    LSTM's inter-layer dropout under checkpointing, where a dropout mask
    that differs on recompute would yield incorrect gradients.
    Checkpointing is skipped when grad is disabled (no graph is built
    then, so there is nothing to save).

    window_init : (B, W, N, F)  -- the true W-window the model was
                                   trained on (the loader's `x`).
    ref_state   : (B, N, F)     -- true next state whose energy is the
                                   drift reference. Defaults to
                                   window_init[:, -1].
    mass        : (B, N)
    """
    if ref_state is None:
        ref_state = window_init[:, -1]
    e0          = total_energy(ref_state[..., :3], ref_state[..., 3:6],
                               mass, eps=eps, g=g)
    denom       = torch.clamp(e0.abs(), min=eps_floor)

    window      = window_init
    drift_accum = window.new_zeros(())        # scalar accumulator
    grad_ctx    = torch.is_grad_enabled()
    for _ in range(K):
        # Full W-window forward (with message passing for the GNN).
        # Checkpointed only for message-passing models under training so
        # K-step BPTT fits in memory: the step's activations are recomputed
        # during backward instead of being held for the whole rollout.
        # MLP/LSTM run directly (cheap, and avoids a recompute-time dropout
        # mask mismatch for the LSTM). use_reentrant=False is required
        # because the window input itself does not require grad --
        # gradients flow to the model parameters, and the non-reentrant
        # path handles that correctly.
        use_ckpt = grad_ctx and getattr(model, "num_passes", None) is not None
        if use_ckpt:
            y_pred = checkpoint(model, window, mass, use_reentrant=False)
        else:
            y_pred = model(window, mass)
        e_pred    = total_energy(y_pred[..., :3], y_pred[..., 3:6],
                                 mass, eps=eps, g=g)
        # Per-sample relative drift then batch mean -- matches the
        # single-step `energy_drift_loss` normalization (ratio of means
        # would weight high-|E0| samples differently than the sibling
        # loss and the docstring promises).
        drift_accum = drift_accum + ((e_pred - e0).abs() / denom).mean()
        # Slide the window: drop the oldest frame, append the prediction
        # (autoregressive). The prediction stays attached so gradients
        # keep flowing through the rollout chain.
        window = torch.cat([window[:, 1:], y_pred.unsqueeze(1)], dim=1)

    return drift_accum / K


# ── Combined loss ────────────────────────────────────────────────────────────
class CombinedLoss(torch.nn.Module):
    """
    Weighted sum of the three losses:

        L = w_mse    · mse_loss(pred, target)
          + w_energy · energy_drift_loss(pred, target, mass, eps, g)
          + w_rollout · rollout_energy_loss(model, window_init, mass,
                                             ref_state, eps, g, K)

    All three terms are differentiable, so the model receives gradients
    from each. Weights are configurable on the CLI via --w-mse,
    --w-energy, --w-rollout.

    Rollout is off by default (w_rollout = 0.0) because K=10 forward
    passes per batch step is a substantial memory cost, turn it on
    once you've confirmed the single-step losses are converging.

    The forward signature matches what your training loops already
    expect: `loss = criterion(pred, target)` plus extra keyword args.
    Since the rollout loss needs the model and the seed window (plus the
    true next state as the drift reference), the training loop calls it
    directly -- see the `w_rollout > 0` branch in any of the
    train_*.py scripts.
    """

    def __init__(self,
                 eps: float = DEFAULT_EPS,
                 g: float = DEFAULT_GRAVITY_G,
                 w_mse: float = 1.0,
                 w_energy: float = 0.1,
                 w_rollout: float = 0.0,
                 rollout_K: int = 10) -> None:
        super().__init__()
        self.eps        = eps
        self.g          = g
        self.w_mse      = w_mse
        self.w_energy   = w_energy
        self.w_rollout  = w_rollout
        self.rollout_K  = rollout_K

    def step_losses(self,
                    pred: torch.Tensor,
                    target: torch.Tensor,
                    mass: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the per-component losses (for logging)."""
        l_mse    = mse_loss(pred, target)
        l_energy = energy_drift_loss(pred, target, mass, eps=self.eps, g=self.g)
        return {"mse": l_mse, "energy": l_energy}

    def rollout(self,
                model: torch.nn.Module,
                window_init: torch.Tensor,
                mass: torch.Tensor,
                ref_state: torch.Tensor | None = None) -> torch.Tensor:
        """K-step sliding-window rollout loss in isolation, for logging."""
        return rollout_energy_loss(model, window_init, mass,
                                   ref_state=ref_state,
                                   eps=self.eps, g=self.g, K=self.rollout_K)

    def __repr__(self) -> str:
        return (f"CombinedLoss(eps={self.eps}, g={self.g}, "
                f"w_mse={self.w_mse}, w_energy={self.w_energy}, "
                f"w_rollout={self.w_rollout}, rollout_K={self.rollout_K})")