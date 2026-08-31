"""
lstm_train.py
=============
Train an LSTM surrogate on the 3D N-body dataset produced by
`3d_export_pipeline.py`. Designed to run end-to-end on Google Colab.

What it does
------------
1. Mounts Google Drive (Colab only: no-op on a local machine).
2. Locates the ML-ready `.npz` archive at
   `<drive_root>/<repo_dir>/ml_ready_data/dataset_3d_w5h1s1r.npz`
   (or whatever `--npz` you point it at).
3. Builds train/val/test `DataLoader`s via `get_dataloaders(model_type="lstm")`.
4. Trains a **per-body LSTM** that maps a single body's W-step state
   sequence (plus mass) to that body's next state. Weights are shared
   across bodies, so the model supports any N at inference.
5. Saves the trained model + loss curve + test metrics to Drive.

Input/output contract
---------------------
With `model_type="lstm"`, each batch is:
    x : (B, W, N*F) : flattened per-body windowed states
    y : (B, N*F)    : flattened next per-body states
The training loop reshapes to (B, W, N, F), appends a per-body mass
channel, and runs the same LSTM on every body independently.

Usage (Colab)
------------
    # 1. Upload the project folder to Drive, e.g.:
    #      /content/drive/MyDrive/Universe-Simulation/
    #    containing at minimum:
    #      3d_pytorch_dataloader.py
    #      lstm_train.py
    #      ml_ready_data/dataset_3d_w5h1s1r.npz
    #      ml_ready_data/dataset_3d_w5h1s1r.json
    #
    # 2. In a Colab cell: use `%run` (not `!python`), because
    #    `drive.mount()` needs the live IPython kernel:
    %run lstm_train.py --epochs 20 --batch-size 64

Usage (local smoke-test)
------------------------
    python lstm_train.py --epochs 2 --batch-size 32 --quick

Notes
-----
- Set the Colab runtime to GPU (Runtime → Change runtime type → T4) for
  a meaningful speedup. CPU works but is slow on the full dataset.
- The LSTM's inductive bias is *temporal* (sequential), in contrast to
  the GNN's *pairwise structural* bias. This is the whole point of the
  three-architecture comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import (
    cap_dataloader,
    configure_utf8_stdout,
    load_sibling_module,
    mount_drive_if_possible,
    pick_device,
    save_loss_curve,
    timestamp,
)

configure_utf8_stdout()

from pipeline_config import (
    DEFAULT_EPS,
    DEFAULT_GRAVITY_G,
    DEFAULT_LR,
    DEFAULT_NPZ,
    DEFAULT_ROLLOUT_K,
    DEFAULT_WEIGHT_DECAY,
    FEATURE_DIM,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    TRAINING_RUNS_DIR,
)

_dl_mod = load_sibling_module("nbody_dataloader", "3d_pytorch_dataloader.py")
get_dataloaders = _dl_mod.get_dataloaders

_loss_mod = load_sibling_module("nbody_losses", "losses.py")
CombinedLoss       = _loss_mod.CombinedLoss
mse_loss           = _loss_mod.mse_loss
energy_drift_loss  = _loss_mod.energy_drift_loss
rollout_energy_loss = _loss_mod.rollout_energy_loss


# ── Model ────────────────────────────────────────────────────────────────────
class LSTMSurrogate(nn.Module):
    """
    Per-body LSTM surrogate.

    Input  layout: (B, W, N, F') where F' = 6 (state) + 1 (mass)
    Output layout: (B, N, 6)

    For each body n we run the same LSTM over its W-step history, using
    the final hidden state to predict that body's next state. This makes
    the model permutation-equivariant and variable-N.
    """

    def __init__(self,
                 window_size: int,
                 in_features: int,
                 hidden: int = LSTM_HIDDEN,
                 num_layers: int = LSTM_LAYERS,
                 dropout: float = LSTM_DROPOUT) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.in_features = int(in_features)
        self.hidden      = int(hidden)
        self.num_layers  = int(num_layers)

        body_in_dim = self.in_features + 1
        self.lstm = nn.LSTM(
            input_size=body_in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,    # input shape (B, W, in_dim)
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.in_features),
        )

    def forward(self, x: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, W, N, F)
        mass : (B, N)
        Returns: (B, N, F)
        """
        B, W, N, F = x.shape
        if W != self.window_size:
            raise ValueError(f"expected window_size={self.window_size}, got {W}")
        if F != self.in_features:
            raise ValueError(f"expected in_features={self.in_features}, got {F}")

        mass_b = mass.view(B, 1, N, 1).expand(B, W, N, 1)
        x_in = torch.cat([x, mass_b], dim=-1)                  # (B, W, N, F+1)
        # Reshape to (B*N, W, F+1) so the same LSTM runs on every body.
        x_seq = x_in.permute(0, 2, 1, 3).reshape(B * N, W, -1)  # (B*N, W, F+1)
        out, _ = self.lstm(x_seq)                               # (B*N, W, hidden)
        last = out[:, -1, :]                                    # (B*N, hidden)
        pred = self.head(last)                                  # (B*N, F)
        return pred.view(B, N, F)

    def step(self, state: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """
        Single-step interface (degenerate W-window).

        Legacy: no live caller. All rollout paths (the rollout-energy
        loss, sweep eval, the stability benchmark, and the OOD runner)
        now seed with the true W-window and slide it, which is
        in-distribution; a degenerate identical-frame window is not.

        state : (B, N, F)
        mass  : (B, N)
        Returns: (B, N, F)
        """
        B, N, F = state.shape
        W = self.window_size
        window = state.unsqueeze(1).expand(B, W, N, F)         # (B, W, N, F)
        return self.forward(window, mass)


# ── Train / eval ─────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs:        int   = LSTM_EPOCHS
    batch_size:    int   = LSTM_BATCH_SIZE
    lr:            float = DEFAULT_LR
    weight_decay:  float = DEFAULT_WEIGHT_DECAY
    hidden:        int   = LSTM_HIDDEN
    num_layers:    int   = LSTM_LAYERS
    dropout:       float = LSTM_DROPOUT
    quick:         bool  = False
    n_bodies:      int | None = None   # optional validation against dataset N
    # Combined-loss weights (see losses.py). Rollout is off by default -
    # it costs K forward passes per batch step.
    w_mse:         float = 1.0
    w_energy:      float = 0.1
    w_rollout:     float = 0.0
    rollout_K:     int   = DEFAULT_ROLLOUT_K
    eps:           float = DEFAULT_EPS
    g:             float = DEFAULT_GRAVITY_G


def _reshape_lstm_batch(x: torch.Tensor,
                        y: torch.Tensor,
                        F: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert LSTM loader flat tensors back to (B, W, N, F).

    x : (B, W, N*F)
    y : (B, N*F)
    Returns: (x: (B, W, N, F), y: (B, N, F), N)
    """
    B, W, NF = x.shape
    N = NF // F
    x_3d = x.view(B, W, N, F)
    y_3d = y.view(B, N, F)
    return x_3d, y_3d, N


def run_epoch(model: nn.Module,
              loader: DataLoader,
              loss_fn: CombinedLoss,
              optimizer: torch.optim.Optimizer | None,
              device: torch.device,
              train: bool,
              F: int) -> dict[str, float]:
    """
    Returns a dict of averaged loss components:
        {mse, energy, rollout, total}
    """
    model.train(train)
    sums  = {"mse": 0.0, "energy": 0.0, "rollout": 0.0, "total": 0.0}
    count = 0
    ctx   = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, mass in loader:
            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            mass = mass.to(device, non_blocking=True)

            x_3d, y_3d, _ = _reshape_lstm_batch(x, y, F=F)
            pred = model(x_3d, mass)                          # (B, N, F)

            l_mse      = mse_loss(pred, y_3d)
            l_energy   = energy_drift_loss(pred, y_3d, mass,
                                           eps=loss_fn.eps, g=loss_fn.g)
            l_total    = (loss_fn.w_mse    * l_mse
                          + loss_fn.w_energy * l_energy)

            if loss_fn.w_rollout > 0.0:
                # In-distribution sliding-window rollout: seed with the
                # true W-window `x_3d` (B, W, N, F) and use the true next
                # state `y_3d` as the energy-drift reference. Matches the
                # sliding-window rollout used in evaluate_models.py and
                # stability_benchmark.py (the earlier model.step path
                # used a degenerate identical-frame window).
                l_roll = rollout_energy_loss(model, x_3d, mass,
                                             ref_state=y_3d,
                                             eps=loss_fn.eps,
                                             g=loss_fn.g,
                                             K=loss_fn.rollout_K)
                l_total = l_total + loss_fn.w_rollout * l_roll
            else:
                l_roll  = pred.new_zeros(())

            if train:
                optimizer.zero_grad(set_to_none=True)
                l_total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bsz = x.size(0)
            sums["mse"]     += float(l_mse.item())    * bsz
            sums["energy"]  += float(l_energy.item()) * bsz
            sums["rollout"] += float(l_roll.item())   * bsz
            sums["total"]   += float(l_total.item())  * bsz
            count          += bsz

    return {k: v / max(count, 1) for k, v in sums.items()}


def _final_test_metrics(model: nn.Module,
                        test_loader: DataLoader,
                        loss_fn: CombinedLoss,
                        device: torch.device,
                        F: int) -> dict[str, float]:
    """Compute per-component test metrics on the held-out test set."""
    model.eval()
    sums  = {"mse": 0.0, "energy": 0.0, "total": 0.0}
    count = 0
    with torch.no_grad():
        for x, y, mass in test_loader:
            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            mass = mass.to(device, non_blocking=True)
            x_3d, y_3d, _ = _reshape_lstm_batch(x, y, F=F)
            pred = model(x_3d, mass)
            l_mse    = mse_loss(pred, y_3d)
            l_energy = energy_drift_loss(pred, y_3d, mass,
                                        eps=loss_fn.eps, g=loss_fn.g)
            l_total  = (loss_fn.w_mse * l_mse
                        + loss_fn.w_energy * l_energy)
            bsz = x.size(0)
            sums["mse"]    += float(l_mse.item())    * bsz
            sums["energy"] += float(l_energy.item()) * bsz
            sums["total"]  += float(l_total.item())  * bsz
            count += bsz
    return {k: v / max(count, 1) for k, v in sums.items()}


def main(cfg: TrainConfig, npz_path: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[device] {device}")

    print(f"[data] {npz_path}")
    train_loader, val_loader, test_loader = get_dataloaders(
        npz_path=npz_path,
        model_type="lstm",
        batch_size=cfg.batch_size,
        include_mass=True,           # mass channel needed for energy loss
        num_workers=2 if device.type == "cuda" else 0,  # 2 workers on GPU: 16GB/4vCPU VM has headroom; mmap dataset shares pages via COW
        pin_memory=(device.type == "cuda"),
    )

    if cfg.quick:
        train_loader = cap_dataloader(train_loader, 8)
        val_loader   = cap_dataloader(val_loader, 8)
        test_loader  = cap_dataloader(test_loader, 2)

    # ── Infer W, N, and F from one batch ─────────────────────────────────
    sample_x, sample_y, sample_mass = next(iter(train_loader))
    B, W, NF = sample_x.shape
    N_bodies = int(sample_y.shape[-1] // FEATURE_DIM)
    in_features = FEATURE_DIM
    if cfg.n_bodies is not None and cfg.n_bodies != N_bodies:
        raise ValueError(
            f"--N {cfg.n_bodies} does not match dataset N={N_bodies}. "
            f"Omit --N to use the dataset's body count."
        )
    print(f"[model] LSTM per-body  W={W}  N={N_bodies}  F={in_features}  "
          f"hidden={cfg.hidden}  num_layers={cfg.num_layers}  "
          f"dropout={cfg.dropout}")

    model     = LSTMSurrogate(
        window_size=W,
        in_features=in_features,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    loss_fn   = CombinedLoss(
        eps=cfg.eps, g=cfg.g,
        w_mse=cfg.w_mse, w_energy=cfg.w_energy,
        w_rollout=cfg.w_rollout, rollout_K=cfg.rollout_K,
    )
    print(f"[loss] {loss_fn}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} trainable parameters")

    history: list[dict] = []
    best_val = float("inf")
    best_path = out_path / "model_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device,
                               train=True, F=in_features)
        val_loss   = run_epoch(model, val_loader,   loss_fn, None,        device,
                               train=False, F=in_features)
        elapsed    = time.perf_counter() - t0

        history.append({
            "epoch":     epoch,
            "train_mse": train_loss["total"],
            "train_components": train_loss,
            "val_mse":   val_loss["total"],
            "val_components":   val_loss,
            "seconds":   elapsed,
        })
        marker = ""
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            # `variant` lets downstream consumers (e.g. stability_benchmark)
            # distinguish the stability-trained checkpoint from the
            # single-step one without relying on the directory name. The
            # heuristic is w_rollout > 0 ⇒ stability-trained.
            variant = "stable" if cfg.w_rollout > 0.0 else "single_step"
            torch.save({
                "model_state":   model.state_dict(),
                "config":        cfg.__dict__,
                "window_size":   W,
                "in_features":   in_features,
                "hidden":        cfg.hidden,
                "num_layers":    cfg.num_layers,
                "epoch":         epoch,
                "val_mse":       val_loss["mse"],
                "val_total":     best_val,
                "variant":       variant,
            }, best_path)
            marker = "  ✓ saved"
        print(f"[epoch {epoch:3d}/{cfg.epochs}] "
              f"train={train_loss['total']:.4e} (mse={train_loss['mse']:.2e} "
              f"energy={train_loss['energy']:.2e} roll={train_loss['rollout']:.2e})  "
              f"val={val_loss['total']:.4e}  ({elapsed:5.2f}s){marker}")

    # ── Final test metrics ──────────────────────────────────────────────────
    print("[test] computing final test metrics...")
    test_metrics = _final_test_metrics(model, test_loader, loss_fn, device,
                                       F=in_features)
    print(f"[test] mse={test_metrics['mse']:.4e}  energy={test_metrics['energy']:.4e}  "
          f"total={test_metrics['total']:.4e}")

    # ── Persist loss history + loss-curve plot ────────────────────────────
    history_path = out_path / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": cfg.__dict__,
            "history": history,
            "test_metrics": test_metrics,
        }, f, indent=2)
    print(f"[save] history  -> {history_path}")
    print(f"[save] best ckpt-> {best_path}  (val_total={best_val:.4e})")

    save_loss_curve(
        history,
        out_path / "loss_curve.png",
        title="LSTM surrogate, per-body recurrent",
        keys=("train_mse", "val_mse"),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a per-body LSTM surrogate on the 3D N-body dataset (Colab-friendly).",
    )
    p.add_argument("--repo-dir", default="",
                   help="Subdirectory of MyDrive containing this project "
                        "(Colab only; leave empty for a local checkout, "
                        "where the repo root is auto-detected).")
    p.add_argument("--npz",      default=None,
                   help="Override the .npz path "
                        "(default: <repo>/ml_ready_data/dataset_3d_w5h1s1r.npz).")
    p.add_argument("--out",      default=None,
                   help="Override the output directory "
                        "(default: <repo>/training_runs/lstm_<ts>).")
    p.add_argument("--epochs",       type=int,   default=LSTM_EPOCHS,
                   help="Number of training epochs.")
    p.add_argument("--N",            type=int,   default=None,
                   help="Optional: validate that the dataset has this many bodies. "
                        "If omitted, N is inferred from the .npz.")
    p.add_argument("--batch-size",   type=int,   default=LSTM_BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--hidden",       type=int,   default=LSTM_HIDDEN)
    p.add_argument("--num-layers",   type=int,   default=LSTM_LAYERS,
                   help="Number of stacked LSTM layers.")
    p.add_argument("--dropout",      type=float, default=LSTM_DROPOUT,
                   help="Inter-layer dropout (only applied when num_layers > 1).")
    p.add_argument("--quick",        action="store_true",
                   help="Smoke-test mode: cap each loader at 8 batches.")
    # ── Combined-loss weights (see losses.py) ───────────────────────────
    p.add_argument("--w-mse",     type=float, default=1.0,
                   help="Weight on per-feature MSE loss.")
    p.add_argument("--w-energy",  type=float, default=0.1,
                   help="Weight on single-step energy-drift loss.")
    p.add_argument("--w-rollout", type=float, default=0.0,
                   help="Weight on K-step autoregressive rollout loss. "
                        "Leave 0 to disable (saves K forward passes per step).")
    p.add_argument("--rollout-K", type=int,   default=DEFAULT_ROLLOUT_K,
                   help="Number of rollout steps when --w-rollout > 0.")
    p.add_argument("--eps",       type=float, default=DEFAULT_EPS,
                   help="Plummer softening for the energy loss.")
    p.add_argument("--g",         type=float, default=DEFAULT_GRAVITY_G,
                   help="Gravitational constant for the energy loss.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    drive_root = mount_drive_if_possible()

    repo_root = Path(drive_root) / args.repo_dir if args.repo_dir else Path(drive_root)
    # Locate the root by checking where the dataset actually lives:
    # on a local checkout `mount_drive_if_possible()` already returns the
    # repo root (appending --repo-dir would double the path), while on
    # Colab the documented layout is <MyDrive>/Universe-Simulation/.
    if not (repo_root / DEFAULT_NPZ).exists():
        if (Path(drive_root) / DEFAULT_NPZ).exists():
            repo_root = Path(drive_root)
        elif (Path(drive_root) / "Universe-Simulation" / DEFAULT_NPZ).exists():
            repo_root = Path(drive_root) / "Universe-Simulation"
    npz_path  = Path(args.npz) if args.npz else repo_root / DEFAULT_NPZ

    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
    else:
        ts = timestamp()
        out_dir = repo_root / TRAINING_RUNS_DIR / f"lstm_{ts}"

    if not npz_path.exists():
        raise SystemExit(
            f"\n[npz] not found: {npz_path}\n"
            f"  Tip: run 3d_export_pipeline.py locally first, then upload the\n"
            f"  entire project folder (including ml_ready_data/) to Drive at\n"
            f"  {repo_root}"
        )

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        quick=args.quick,
        n_bodies=args.N,
        w_mse=args.w_mse,
        w_energy=args.w_energy,
        w_rollout=args.w_rollout,
        rollout_K=args.rollout_K,
        eps=args.eps,
        g=args.g,
    )
    print(f"[run] cfg={cfg}")
    print(f"[run] npz ={npz_path}")
    print(f"[run] out ={out_dir}")
    main(cfg, npz_path, out_dir)
