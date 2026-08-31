"""
gnn_train.py
============
Train a Graph Neural Network (GNN) surrogate on the 3D N-body dataset
produced by `3d_export_pipeline.py`. Designed to run end-to-end on Google
Colab.

What it does
------------
1. Mounts Google Drive (Colab only: no-op on a local machine).
2. Locates the ML-ready `.npz` archive at
   `<drive_root>/<repo_dir>/ml_ready_data/dataset_3d_w5h1s1r.npz`
   (or whatever `--npz` you point it at).
3. Builds train/val `DataLoader`s via `get_dataloaders(model_type="gnn")`
  - the loader reshapes each window to `(W, N, F)` so the GNN sees the
   per-body state at every timestep.
4. Trains a small message-passing GNN where each timestep runs a round
   of message passing over fully connected body nodes; the final
   timestep's node embeddings feed a per-node readout MLP that predicts
   each body's next state.
5. Saves the trained model + a loss curve to Drive so you can pull them
   back.

Why a custom GNN instead of torch_geometric
-------------------------------------------
- No extra dependency to install on Colab.
- The graph here is fully connected and fixed (every body interacts
  with every other), so the message-passing loop collapses to a
  vectorised pairwise sum: the same O(N²) pattern the upstream
  `simulation_3d.compute_accelerations` uses, but learned.
- Operates on plain dense tensors, which fits the dataloader's
  `(B, W, N, F)` layout without any graph-batching machinery.

Input/output contract (set by the dataloader)
--------------------------------------------
With `model_type="gnn"`, each batch is:
    x : (B, W, N, F) : windowed per-body per-timestep states
    y : (B, N, F)    : next per-body state

Usage (Colab)
------------
    # 1. Upload the project folder to Drive, e.g.:
    #      /content/drive/MyDrive/Universe-Simulation/
    #    containing at minimum:
    #      3d_pytorch_dataloader.py
    #      gnn_train.py
    #      ml_ready_data/dataset_3d_w5h1s1r.npz
    #      ml_ready_data/dataset_3d_w5h1s1r.json
    #
    # 2. In a Colab cell: use `%run` (not `!python`), because
    #    `drive.mount()` needs the live IPython kernel:
    %run gnn_train.py --epochs 20 --batch-size 32

Usage (local smoke-test)
------------------------
    python gnn_train.py --epochs 2 --batch-size 8 --quick

Notes
-----
- Set the Colab runtime to GPU (Runtime → Change runtime type → T4) for
  a meaningful speedup. CPU works but is slow on the full dataset.
- The GNN's inductive bias is *pairwise structural* (every body
  exchanges messages with every other). This is the same physics-aligned
  inductive bias that motivates Sanchez-Gonzalez et al. (2020),
  Satorras et al. (2021), and the rest of the GNN-for-physics-sim
  literature cited in the project.
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
    GNN_BATCH_SIZE,
    GNN_EPOCHS,
    GNN_HIDDEN,
    GNN_PASSING_STEPS,
    TRAINING_RUNS_DIR,
)

_dl_mod = load_sibling_module("nbody_dataloader", "3d_pytorch_dataloader.py")
get_dataloaders = _dl_mod.get_dataloaders

_loss_mod = load_sibling_module("nbody_losses", "losses.py")
CombinedLoss        = _loss_mod.CombinedLoss
mse_loss            = _loss_mod.mse_loss
energy_drift_loss   = _loss_mod.energy_drift_loss
rollout_energy_loss = _loss_mod.rollout_energy_loss


# ── Model ────────────────────────────────────────────────────────────────────
class GNNSurrogate(nn.Module):
    """
    A small message-passing GNN over fully connected body nodes.

    Per timestep:
        1. Embed raw node features (x, y, z, vx, vy, vz) → `hidden`
        2. For `num_message_passes` rounds:
              m_ij = message_mlp([h_i, h_j − h_i, ‖r_j − r_i‖])
              m_i  = Σ_j m_ij              (sum aggregation)
              h_i  = GRU_cell(h_i, m_i)     (recurrent node update)
        3. After the timestep, run the next timestep with the updated
           embeddings. Repeat for all W timesteps.
        4. After the W-th timestep, pass each node's final embedding
           through a per-node readout MLP to predict that body's next
           state.

    Vectorisation
    -------------
    With N nodes per graph and B graphs in a batch, the pairwise
    message tensor has shape (B, N, N, 3*hidden), that's the same
    memory footprint as the upstream `compute_accelerations`, just
    bigger because the message MLP has more parameters than Newton's
    inverse-square law. For the default 3D pipeline (B=32, N=25, h=128)
    this is ~3 MB per batch, well within GPU memory.

    Parameters
    ----------
    in_features     : F (kept features per body, typically 6). The encoder
                      internally appends a +1 mass channel.
    hidden          : message-passing hidden size
    num_passes      : message-passing rounds per timestep
    """

    def __init__(self,
                 in_features: int,
                 hidden: int = GNN_HIDDEN,
                 num_passes: int = GNN_PASSING_STEPS) -> None:
        super().__init__()
        self.hidden      = hidden
        self.num_passes  = num_passes

        # The forward pass always appends a per-body mass channel, so the
        # encoder must take in_features + 1 inputs (state + mass). Pass
        # `in_features=6` from the loader and we'll add the +1 internally.
        enc_in = in_features + 1

        # ── Node encoder: raw features → hidden ────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        # ── Edge feature extractor: takes [h_i, h_j − h_i, ‖r‖] ────────────
        # We split position from velocity in the input to extract ‖r‖.
        # Caller supplies the position slice (first 3 channels of F).
        # The MLP input is therefore:
        #   2 * hidden   (h_i and h_j - h_i)
        #   + 1          (scalar distance)
        edge_in_dim = 2 * hidden + 1
        self.message_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )

        # ── Node update: GRU cell, stable against long message-passing chains ─
        self.update_cell = nn.GRUCell(hidden, hidden)

        # ── Per-node readout: hidden → next-state features ─────────────────
        # Outputs enc_in channels (state + mass), but forward slices off
        # the last channel so callers see a clean (B, N, in_features).
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, enc_in),
        )

    @staticmethod
    def _pairwise_features(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the per-edge feature tensors used by the message MLP.

        h : (B, N, hidden)  node embeddings
        Returns
            diff  : (B, N, N, hidden)  h_j - h_i  for every pair (i, j)
            delta : (B, N, N, 1)       ‖(h_j - h_i)‖  scalar distance proxy
        """
        # h_j - h_i via broadcasting, same trick simulation_3d uses for forces.
        diff  = h.unsqueeze(2) - h.unsqueeze(1)            # (B, N, N, hidden)
        delta = diff.norm(dim=-1, keepdim=True)            # (B, N, N, 1)
        return diff, delta

    def forward(self, x: torch.Tensor, mass: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : (B, W, N, F) : windowed per-body states (F = 6 by default)
        mass : (B, N)       : per-body mass channel broadcast across the W
                              timesteps. Required: the GNN encoder always
                              reads `in_features + 1` channels (state + mass)
                              so the energy loss can reconstruct the same
                              per-body features the model saw during training.

        Returns (B, N, F): predicted next per-body state.

        The loader's gnn mode always supplies `mass`; the no-mass branch
        has been removed to keep the channel bookkeeping unambiguous.
        """
        assert mass is not None, "GNN requires mass for now"
        B, W, N, F  = x.shape
        mass_b      = mass.view(B, 1, N, 1).expand(B, W, N, 1)
        x           = torch.cat([x, mass_b], dim=-1)         # (B, W, N, F+1)
        B, W, N, Fp = x.shape

        # Iterate over timesteps. We embed each timestep's nodes, run
        # message passing, and carry the updated embeddings forward.
        h = self.node_encoder(x[:, 0, :, :])                # (B, N, hidden)
        for t in range(1, W):
            h_t = self.node_encoder(x[:, t, :, :])         # (B, N, hidden)

            # Message passing: combine the per-step embedding with the
            # carried-forward state by *summing* their hidden vectors
            # before each round of messaging. This is one of the simpler
            # "skip" connection strategies, empirically a sum works as
            # well as a concat here.
            h_combined = h_t + h                           # (B, N, hidden)
            for _ in range(self.num_passes):
                diff, delta = self._pairwise_features(h_combined)
                # Concatenate the edge features along the channel axis.
                edge_feat = torch.cat([h_combined.unsqueeze(2).expand(-1, -1, N, -1),
                                       diff,
                                       delta], dim=-1)     # (B, N, N, 2h+1)
                messages  = self.message_mlp(edge_feat)    # (B, N, N, hidden)
                # Aggregate by sum over the source index.
                agg       = messages.sum(dim=2)            # (B, N, hidden)
                h         = self.update_cell(
                    agg.reshape(-1, self.hidden),
                    h.reshape(-1, self.hidden),
                ).reshape(B, N, self.hidden)

        # Per-node readout from the final hidden state.
        # The encoder was fed F+1 channels (state + mass); the readout
        # outputs F+1 channels too. We return only the first F (state)
        # so the loss functions see a clean (B, N, 6) tensor.
        out = self.readout(h)                              # (B, N, F+1)
        return out[..., :Fp - 1]

    def step(self, state: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """
        Legacy single-step interface (W=1).

        Wraps `state` as a 1-step window and calls `forward`. Kept for
        reference only -- it is NOT used by the rollout-energy loss or by
        any evaluation path, because with W=1 the message-passing loop
        `for t in range(1, W)` in `forward` runs *zero* rounds: the GNN
        would predict with no pairwise messages, an out-of-distribution
        path that previously corrupted the stability-trained and OOD GNN
        results. The rollout loss (`losses.rollout_energy_loss`) and all
        evaluators now call the full W-window `forward` directly.

        state : (B, N, F)
        mass  : (B, N)
        Returns: (B, N, F)
        """
        x = state.unsqueeze(1)                             # (B, 1, N, F)
        return self.forward(x, mass=mass)                  # (B, N, F)


# ── Train / eval ─────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs:        int   = GNN_EPOCHS
    batch_size:    int   = GNN_BATCH_SIZE
    lr:            float = DEFAULT_LR
    weight_decay:  float = DEFAULT_WEIGHT_DECAY
    hidden:        int   = GNN_HIDDEN
    num_passes:    int   = GNN_PASSING_STEPS
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


def run_epoch(model: nn.Module,
              loader: DataLoader,
              loss_fn: CombinedLoss,
              optimizer: torch.optim.Optimizer | None,
              device: torch.device,
              train: bool) -> dict[str, float]:
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

            # GNN input: (B, W, N, F): mass is appended inside the model.
            pred = model(x, mass=mass)                       # (B, N, 6)

            l_mse      = mse_loss(pred, y)
            l_energy   = energy_drift_loss(pred, y, mass,
                                           eps=loss_fn.eps, g=loss_fn.g)
            l_total    = (loss_fn.w_mse    * l_mse
                          + loss_fn.w_energy * l_energy)

            if loss_fn.w_rollout > 0.0:
                # In-distribution sliding-window rollout: seed with the
                # true W-window `x` (B, W, N, F) so the GNN's forward
                # runs its full message-passing path, and use the true
                # next state `y` as the energy-drift reference. This
                # matches how the GNN is evaluated in evaluate_models.py
                # and stability_benchmark.py. (An earlier version called
                # model.step here, which wraps the state as a W=1 window
                # and so ran zero message-passing rounds -- training the
                # stability term on an out-of-distribution path.)
                l_roll = rollout_energy_loss(model, x, mass,
                                             ref_state=y,
                                             eps=loss_fn.eps,
                                             g=loss_fn.g,
                                             K=loss_fn.rollout_K)
                l_total = l_total + loss_fn.w_rollout * l_roll
            else:
                l_roll  = pred.new_zeros(())

            if train:
                optimizer.zero_grad(set_to_none=True)
                l_total.backward()
                # Gradient clipping, message-passing GNNs can have
                # sharp gradient magnitudes when many passes accumulate.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bsz = x.size(0)
            sums["mse"]     += float(l_mse.item())    * bsz
            sums["energy"]  += float(l_energy.item()) * bsz
            sums["rollout"] += float(l_roll.item())   * bsz
            sums["total"]   += float(l_total.item())  * bsz
            count          += bsz

    return {k: v / max(count, 1) for k, v in sums.items()}


def main(cfg: TrainConfig, npz_path: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[device] {device}")

    print(f"[data] {npz_path}")
    train_loader, val_loader, test_loader = get_dataloaders(
        npz_path=npz_path,
        model_type="gnn",
        batch_size=cfg.batch_size,
        include_mass=True,           # mass channel needed for energy loss
        num_workers=2 if device.type == "cuda" else 0,  # 2 workers on GPU: 16GB/4vCPU VM has headroom; mmap dataset shares pages via COW
        pin_memory=(device.type == "cuda"),
    )

    if cfg.quick:
        train_loader = cap_dataloader(train_loader, 8)
        val_loader   = cap_dataloader(val_loader, 8)
        test_loader  = cap_dataloader(test_loader, 2)

    # ── Infer N and F from one batch ───────────────────────────────────────
    sample_x, sample_y, _ = next(iter(train_loader))
    in_features = int(sample_x.shape[-1])
    N_bodies = int(sample_x.shape[-2])
    if cfg.n_bodies is not None and cfg.n_bodies != N_bodies:
        raise ValueError(
            f"--N {cfg.n_bodies} does not match dataset N={N_bodies}. "
            f"Omit --N to use the dataset's body count."
        )
    print(f"[model] GNN  N={N_bodies}  in_features={in_features}  "
          f"hidden={cfg.hidden}  num_passes={cfg.num_passes}")

    model     = GNNSurrogate(
        in_features=in_features,
        hidden=cfg.hidden,
        num_passes=cfg.num_passes,
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
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   loss_fn, None,        device, train=False)
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
                "in_features":   in_features,
                "hidden":        cfg.hidden,    # for evaluate_models.py
                "num_passes":    cfg.num_passes,
                "epoch":         epoch,
                "val_mse":       val_loss["total"],
                "variant":       variant,
            }, best_path)
            marker = "  ✓ saved"
        print(f"[epoch {epoch:3d}/{cfg.epochs}] "
              f"train={train_loss['total']:.4e} (mse={train_loss['mse']:.2e} "
              f"energy={train_loss['energy']:.2e} roll={train_loss['rollout']:.2e})  "
              f"val={val_loss['total']:.4e}  ({elapsed:5.2f}s){marker}")

    # ── Final test metrics ──────────────────────────────────────────────────
    print("[test] computing final test metrics...")
    model.eval()
    test_sums = {"mse": 0.0, "energy": 0.0, "total": 0.0}
    test_count = 0
    with torch.no_grad():
        for x, y, mass in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mass = mass.to(device, non_blocking=True)
            pred = model(x, mass=mass)
            l_mse = mse_loss(pred, y)
            l_energy = energy_drift_loss(pred, y, mass,
                                         eps=loss_fn.eps, g=loss_fn.g)
            l_total = (loss_fn.w_mse * l_mse
                       + loss_fn.w_energy * l_energy)
            bsz = x.size(0)
            test_sums["mse"] += float(l_mse.item()) * bsz
            test_sums["energy"] += float(l_energy.item()) * bsz
            test_sums["total"] += float(l_total.item()) * bsz
            test_count += bsz
    test_metrics = {k: v / max(test_count, 1) for k, v in test_sums.items()}
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
        title="GNN surrogate, message passing on body graph",
        keys=("train_mse", "val_mse"),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a GNN surrogate on the 3D N-body dataset (Colab-friendly).",
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
                        "(default: <repo>/training_runs/gnn_<ts>).")
    p.add_argument("--epochs",       type=int,   default=GNN_EPOCHS,
                   help="Number of training epochs.")
    p.add_argument("--N",            type=int,   default=None,
                   help="Optional: validate that the dataset has this many bodies. "
                        "If omitted, N is inferred from the .npz.")
    p.add_argument("--batch-size",   type=int,   default=GNN_BATCH_SIZE,
                   help="Smaller batches than MLP/LSTM, the (B,N,N,hidden) "
                        "message tensor is memory-hungry.")
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--hidden",       type=int,   default=GNN_HIDDEN)
    p.add_argument("--num-passes",   type=int,   default=GNN_PASSING_STEPS,
                   help="Number of message-passing rounds per timestep.")
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
        out_dir = repo_root / TRAINING_RUNS_DIR / f"gnn_{ts}"

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
        num_passes=args.num_passes,
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