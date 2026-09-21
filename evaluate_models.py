"""
evaluate_models.py
==================
Unified benchmark script for trained neural surrogates on the 3D N-body
dataset. Produces the four metrics the project comparison rests on:

    1. MSE           : single-step positional / velocity MSE on the
                        requested split (default test)
    2. Latency       , mean forward-pass wall time per batch (ms/batch,
                        and per-sample μs/sample) measured under `no_grad`
    3. Energy error  , |E(pred) − E(true)| / |E(true)| averaged over the
                        split (single-step)
    4. Rollout stability
                    - K-step autoregressive rollout energy drift relative
                        to the initial energy (mean over K steps, averaged
                        over multiple batches). Lower is more stable.

Inputs
------
- One or more checkpoint `.pt` files. Each checkpoint must include the
  keys written by `mlp_train.py` / `lstm_train.py` / `gnn_train.py`:
      model_state, config, window_size/in_features/state_dim
  The script also accepts a CLI `--model-type` per checkpoint (default
  `gnn`) so it can load MLP, LSTM, or GNN weights into the right
  architecture.

Output
------
- A markdown table printed to stdout (and saved to `metrics.json`).
- A bar chart per metric, saved to `plots/eval_<metric>.png`.

Usage
-----
    # Compare three checkpoints on the test split
    python evaluate_models.py \\
        --ckpt runs/gnn/model_best.pt:gnn \\
        --ckpt runs/lstm/model_best.pt:lstm \\
        --ckpt runs/mlp/model_best.pt:mlp \\
        --npz ml_ready_data/dataset_3d_w5h1s1r.npz \\
        --split test --rollout-K 50 --rollout-batches 4

Notes
-----
- By default all metrics run on the *test* split, never train.
- The script uses `torch.no_grad()` for latency and energy-error timing
  (no gradient bookkeeping). Rollout stability is evaluated in `eval`
  mode with autoregressive detaching.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from utils import (
    configure_utf8_stdout,
    load_checkpoint,
    load_sibling_module,
    pick_device,
)

configure_utf8_stdout()

from pipeline_config import (
    DEFAULT_NPZ,
    DEFAULT_ROLLOUT_K,
    EVAL_BATCH_SIZE,
    FEATURE_DIM,
    ModelType,
    PLOTS_DIR,
)


# ── Local imports ─────────────────────────────────────────────────────────────
_loss_mod = load_sibling_module("nbody_losses", "losses.py")
_dl_mod   = load_sibling_module("nbody_dataloader", "3d_pytorch_dataloader.py")
get_dataloaders        = _dl_mod.get_dataloaders
mse_loss               = _loss_mod.mse_loss
energy_drift_loss      = _loss_mod.energy_drift_loss
rollout_energy_loss    = _loss_mod.rollout_energy_loss
total_energy           = _loss_mod.total_energy

_MLP_MOD  = load_sibling_module("_eval_mlp", "mlp_train.py")
_LSTM_MOD = load_sibling_module("_eval_lstm", "lstm_train.py")
_GNN_MOD  = load_sibling_module("_eval_gnn", "gnn_train.py")

MLPSurrogate  = _MLP_MOD.MLPSurrogate
LSTMSurrogate = _LSTM_MOD.LSTMSurrogate
GNNSurrogate  = _GNN_MOD.GNNSurrogate


# ── Build a model from a checkpoint ──────────────────────────────────────────
def build_model(ckpt_path: str, model_type: str,
                device: torch.device) -> torch.nn.Module:
    """
    Load `ckpt_path`, infer the architecture from the checkpoint's saved
    config (or fall back to the `hidden`/`num_layers`/`num_passes` keys
    the trainers now write), then build the matching `*Surrogate` class
    and load the state dict.

    All three trainers now save:
        window_size / in_features  (MLP, LSTM)
        in_features                (GNN)
        hidden, depth/num_layers/num_passes
    so no CLI override is needed for IO dimensions.
    """
    ckpt = load_checkpoint(ckpt_path)

    cfg_dict = ckpt.get("config", {}) or {}
    hidden     = ckpt.get("hidden",     cfg_dict.get("hidden",     128))
    depth      = ckpt.get("depth",      cfg_dict.get("depth",      4))
    num_layers = ckpt.get("num_layers", cfg_dict.get("num_layers", 2))
    num_passes = ckpt.get("num_passes", cfg_dict.get("num_passes", 2))

    if model_type == "mlp":
        window_size = ckpt.get("window_size",
                               cfg_dict.get("window_size", 5))
        in_features = ckpt.get("in_features",
                               cfg_dict.get("in_features", FEATURE_DIM))
        model = MLPSurrogate(window_size=window_size,
                             in_features=in_features,
                             hidden=hidden,
                             depth=depth)
    elif model_type == "lstm":
        window_size = ckpt.get("window_size",
                               cfg_dict.get("window_size", 5))
        in_features = ckpt.get("in_features",
                               cfg_dict.get("in_features", FEATURE_DIM))
        model = LSTMSurrogate(window_size=window_size,
                              in_features=in_features,
                              hidden=hidden,
                              num_layers=num_layers)
    elif model_type == "gnn":
        in_features = ckpt.get("in_features",
                               cfg_dict.get("in_features", FEATURE_DIM))
        model = GNNSurrogate(in_features=in_features,
                             hidden=hidden,
                             num_passes=num_passes)
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


# ── Metrics ──────────────────────────────────────────────────────────────────
@dataclass
class Metrics:
    model_name: str
    model_type: str
    n_params:    int
    split:       str
    mse:         float   # single-step MSE
    energy:      float   # single-step |ΔE/E₀|
    latency_ms:  float   # mean forward-pass latency (ms/batch)
    rollout:     float   # K-step rollout energy drift
    n_samples:   int
    n_batches:   int
    rollout_K:   int
    rollout_batches: int
    # Identity (persistence) baseline: MSE of predicting the last input
    # frame as the next state, on the SAME split. A trained model must
    # beat this floor; explained variance = 1 − mse/mse_identity makes
    # the margin scale-free (1.0 = perfect, 0.0 = identity, <0 = worse
    # than identity). Permanent audit column, 2026-09-21: catches the
    # identity-collapse failure mode (the Sept-15 retrain's root cause)
    # at eval time, not just at the training gate.
    mse_identity:      float = float("nan")
    mse_explained_var: float = float("nan")


def _predict_one_step(model: torch.nn.Module, model_type: str,
                      x: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Returns pred of shape (B, N, FEATURE_DIM)."""
    if model_type == "mlp":
        # Loader returns x as (B, W*N*F); reshape to (B, W, N, F).
        B = x.shape[0]
        # Recover N from mass and W from the flat size.
        N = mass.shape[-1]
        W = model.window_size
        F = model.in_features
        x_3d = x.view(B, W, N, F)
        return model(x_3d, mass)
    elif model_type == "lstm":
        # Loader returns x as (B, W, N*F).
        B, W, NF = x.shape
        N = mass.shape[-1]
        F = model.in_features
        x_3d = x.view(B, W, N, F)
        return model(x_3d, mass)
    elif model_type == "gnn":
        return model(x, mass=mass)
    else:
        raise ValueError(model_type)


def evaluate(model: torch.nn.Module,
             model_type: str,
             loader,
             split_name: str,
             device: torch.device,
             K: int = 50,
             rollout_batches: int = 1) -> Metrics:
    model.eval()

    mse_sum, energy_sum = 0.0, 0.0
    ident_sum = 0.0
    n_samples = 0

    latencies: list[float] = []
    n_timed = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(batch) == 3:
                x, y, mass = batch
            else:
                x, y = batch
                mass = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mass is not None:
                mass = mass.to(device, non_blocking=True)

            pred = _predict_one_step(model, model_type, x, mass)
            # Ensure y is (B, N, F).
            if model_type in ("mlp", "lstm"):
                N = mass.shape[-1]
                y = y.view(-1, N, FEATURE_DIM)

            mse_val = float(mse_loss(pred, y).item())
            if mass is not None:
                e_val = float(energy_drift_loss(pred, y, mass).item())
            else:
                e_val = float("nan")

            # Identity (persistence) baseline on the same batch: repeat the
            # last input frame as the prediction. Free (no forward pass) and
            # measured on exactly the same (x, y) pairs as the model, so the
            # comparison is like-for-like.
            if model_type == "gnn":
                x_last = x[:, -1, :, :]                      # (B, N, F)
            else:
                B = x.shape[0]
                W = model.window_size
                N = mass.shape[-1]
                F = model.in_features
                x_last = x.view(B, W, N, F)[:, -1, :, :]     # (B, N, F)
            ident_val = float(mse_loss(x_last, y).item())

            # Latency timing (skip the first batch as warmup).
            if i > 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = _predict_one_step(model, model_type, x, mass)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
                n_timed += 1

            bsz = x.size(0)
            mse_sum   += mse_val  * bsz
            energy_sum += e_val   * bsz
            ident_sum += ident_val * bsz
            n_samples += bsz

    mse_avg    = mse_sum   / max(n_samples, 1)
    energy_avg = energy_sum / max(n_samples, 1)
    ident_avg  = ident_sum / max(n_samples, 1)
    # Explained variance vs identity: >0 means the model beats the
    # persistence floor on this split. Guard: if the identity MSE is ~0 the
    # targets are ~constant and the ratio is meaningless → NaN.
    if ident_avg > 0.0 and np.isfinite(ident_avg):
        ev = 1.0 - mse_avg / ident_avg
    else:
        ev = float("nan")
    latency_ms = float(statistics.fmean(latencies)) if latencies else float("nan")

    # ── Rollout stability (average over the first N rollout batches) ───────
    rollout = float("nan")
    if K > 0 and rollout_batches > 0:
        drift_total = 0.0
        n_roll = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= rollout_batches:
                    break
                if len(batch) == 3:
                    x, y, mass = batch
                else:
                    _, y = batch
                    mass = None
                if mass is None:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mass = mass.to(device, non_blocking=True)
                if model_type in ("mlp", "lstm"):
                    N = mass.shape[-1]
                    y = y.view(-1, N, FEATURE_DIM)
                # Per-sample denominator: the metric is the MEAN OF PER-SAMPLE
                # RATIOS mean(|ΔE|/|E₀|), matching the single-step energy
                # metric above, `real_case_validation/metrics.py` and
                # `stability_benchmark.py`. An earlier version divided
                # mean(|ΔE|) by mean(|E₀|) (ratio-of-means), which is not the
                # same quantity and made the rollout column incomparable to
                # every other energy number (audited fix, 2026-09-14).
                N = mass.shape[-1]
                # Build the (B, W, N, F) input window the rollout will
                # slide. The MLP/LSTM loaders return flat tensors so we
                # reshape them here; the GNN loader already returns
                # (B, W, N, F).
                if model_type == "mlp":
                    B = x.shape[0]
                    W = model.window_size
                    F = model.in_features
                    window = x.view(B, W, N, F)              # (B, W, N, F)
                elif model_type == "lstm":
                    B, W, _NF = x.shape
                    F = model.in_features
                    window = x.view(B, W, N, F)              # (B, W, N, F)
                else:  # gnn: x is already (B, W, N, F)
                    window = x
                # Anchor the rollout's E0 to the LAST FRAME OF THE INPUT
                # WINDOW (the model's starting point), NOT the target
                # frame `y`. The earlier version anchored to `y` and
                # produced a metric of `|E(pred_k) - E(target)| /
                # |E(target)|`, which is not the autoregressive rollout
                # drift — it's how far the predictions drift from the
                # ground-truth next state, an entirely different quantity.
                # This is the fix that makes the K-step rollout column
                # comparable to `stability_benchmark.py`'s energy_drift
                # and `losses.rollout_energy_loss`'s reference (both anchor
                # to the rollout's start frame).
                e0 = total_energy(window[:, -1, :, :3],
                                  window[:, -1, :, 3:6], mass)
                denom = e0.abs().clamp(min=1e-8)             # (B,)
                drift_acc = 0.0
                # In-distribution sliding W-window rollout for every model
                # (matches stability_benchmark.rollout_sliding and the
                # rollout-energy loss in losses.py). All three surrogates were
                # trained on W=5 windows, so each is seeded with the loader's
                # true W-window and the window is shifted with each prediction.
                # An earlier version used `model.step` (a degenerate W=1 /
                # identical-frame window) for MLP/LSTM, which is
                # out-of-distribution and inconsistent with the GNN branch and
                # the stability benchmark; it is now unified on the sliding
                # window so the tab:eval K-step rollout column and the
                # stability benchmark measure the same path.
                for _ in range(K):
                    pred = model(window, mass)               # (B, N, F)
                    e = total_energy(pred[..., :3], pred[..., 3:6], mass)
                    drift_acc += ((e - e0).abs() / denom).mean().item()
                    window = torch.cat([window[:, 1:], pred.unsqueeze(1)], dim=1)
                drift_total += drift_acc / max(K, 1)
                n_roll += 1
        if n_roll > 0:
            rollout = drift_total / max(n_roll, 1)

    n_params = sum(p.numel() for p in model.parameters())
    return Metrics(
        model_name=model.__class__.__name__,
        model_type=model_type,
        n_params=n_params,
        split=split_name,
        mse=mse_avg,
        energy=energy_avg,
        latency_ms=latency_ms,
        rollout=rollout,
        n_samples=n_samples,
        n_batches=n_timed,
        rollout_K=K,
        rollout_batches=rollout_batches,
        mse_identity=ident_avg,
        mse_explained_var=ev,
    )


# ── Pretty-print ─────────────────────────────────────────────────────────────
def format_metrics_table(metrics_list: list[Metrics]) -> str:
    header = (f"{'model':<22} {'type':<5} {'split':<6} {'#params':>10}  "
              f"{'MSE':>10}  {'|ΔE/E₀|':>10}  "
              f"{'latency(ms)':>12}  {'rollout-K':>10}  {'rollout':>10}  "
              f"{'MSE_ident':>10}  {'expl.var':>8}")
    lines = [header, "-" * len(header)]
    for m in metrics_list:
        lines.append(
            f"{m.model_name:<22} {m.model_type:<5} {m.split:<6} {m.n_params:>10,}  "
            f"{m.mse:>10.3e}  {m.energy:>10.3e}  "
            f"{m.latency_ms:>12.3f}  {m.rollout_K:>10d}  "
            f"{m.rollout:>10.3e}  {m.mse_identity:>10.3e}  "
            f"{m.mse_explained_var:>8.3f}"
        )
    return "\n".join(lines)


def make_plots(metrics_list: list[Metrics], out_dir: str) -> None:
    """One PNG per metric, plus a side-by-side bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: {e!r}")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    names = [f"{m.model_name}\n({m.model_type})" for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    titles = ["MSE (↓ better)", "|ΔE/E₀| single-step (↓ better)",
              "Latency ms/batch (↓ better)",
              f"Rollout stability (↓ better, K={metrics_list[0].rollout_K})"]
    values = [
        [m.mse      for m in metrics_list],
        [m.energy   for m in metrics_list],
        [m.latency_ms for m in metrics_list],
        [m.rollout  for m in metrics_list],
    ]
    for ax, t, v in zip(axes.flat, titles, values):
        ax.bar(names, v)
        ax.set_title(t)
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.suptitle("3D N-body surrogate, four-metric benchmark",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plot_path = out_path / "eval_benchmark.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[plot] {plot_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_ckpt(spec: str) -> tuple[str, str]:
    """`path/to.pt:mlp` → ('path/to.pt', 'mlp'). Default type: gnn."""
    if ":" in spec:
        path, kind = spec.rsplit(":", 1)
        kind = kind.strip().lower()
        if kind not in ModelType.values():
            raise ValueError(f"unknown model_type {kind!r}; expected one of {ModelType.values()}")
        return path, kind
    return spec, ModelType.GNN


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark trained MLP/LSTM/GNN surrogates on 4 project metrics."
    )
    p.add_argument("--ckpt", action="append", default=None,
                   help="Checkpoint spec 'path.pt:model_type'. Repeatable. "
                        "If omitted, every sweep checkpoint found under "
                        "training_runs/N*/{mlp,lstm,gnn}/model_best.pt is "
                        "evaluated against its matching w{N} dataset.")
    p.add_argument("--npz", default=DEFAULT_NPZ,
                   help="Dataset archive.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Which persisted split to evaluate on.")
    p.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    p.add_argument("--rollout-K", type=int, default=DEFAULT_ROLLOUT_K,
                   help="Number of autoregressive rollout steps for stability.")
    p.add_argument("--rollout-batches", type=int, default=1,
                   help="Average rollout stability over this many batches.")
    p.add_argument("--out", default=PLOTS_DIR,
                   help="Directory to write the bar chart PNG.")
    p.add_argument("--json", default=None,
                   help="Optional JSON path to dump raw metrics.")
    args = p.parse_args()

    device = pick_device()
    print(f"[device] {device}")

    # Resolve checkpoint groups. Each group = (npz_path, [(ckpt, type), ...]).
    if args.ckpt:
        groups: list[tuple[str, list[str]]] = [(args.npz, args.ckpt)]
    else:
        groups = []
        sweep_dirs = sorted(Path("training_runs").glob("N*"),
                            key=lambda p: (len(p.name), p.name))
        for n_dir in sweep_dirs:
            found: list[str] = []
            # Single-step ckpts: training_runs/N{N}/{model}/model_best.pt
            # Stable ckpts:    training_runs/N{N}/{model}_stable/model_best.pt
            for m in ModelType.values():
                for variant_dir in (m, f"{m}_stable"):
                    cand = n_dir / variant_dir / "model_best.pt"
                    if cand.is_file():
                        found.append(f"{cand}:{m}")
            if found:
                # Filename is window-annotated (W=5, H=1, S=1, raw norm),
                # NOT body-annotated. The body count N lives in the
                # directory name, not the file. Use the constant
                # w5h1s1r suffix for every N; an earlier version
                # interpolated the body count into the filename and
                # produced nonexistent paths like `w10h1s1r.npz`.
                # Each ckpt's path encodes its model type as the second
                # to-last component (`.../{model}/...` or
                # `.../{model}_stable/...`); use that to point at the
                # matching dataset.
                for ckpt_spec in found:
                    ckpt_path = ckpt_spec.split(":", 1)[0]
                    model_type = ckpt_path.split("/")[-2].rstrip("_stable")
                    npz_path = (Path("ml_ready_data") / f"N{n_dir.name[1:]}"
                                / model_type / "dataset_3d_w5h1s1r.npz")
                    groups.append((str(npz_path), [ckpt_spec]))
        if not groups:
            raise SystemExit(
                "\n[eval] no checkpoints given and none found under "
                "training_runs/N*/{mlp,lstm,gnn}/model_best.pt.\n"
                "  Train first (scaling_sweep.py or the *_train.py scripts)\n"
                "  or pass explicit --ckpt path.pt:model_type specs.")

    metrics_list: list[Metrics] = []
    for npz_group, specs in groups:
        for spec in specs:
            try:
                ckpt_path, model_type = _parse_ckpt(spec)
            except ValueError as e:
                raise SystemExit(f"\n[eval] bad --ckpt spec {spec!r}: {e}\n"
                                 "  Expected 'path.pt:model_type' with "
                                 f"model_type in {ModelType.values()}.")
            print(f"\n[eval] {model_type:5s} <- {ckpt_path}")

            # Build all three loaders, then pick the requested split.
            try:
                train_loader, val_loader, test_loader = get_dataloaders(
                    npz_path=npz_group,
                    model_type=model_type,
                    batch_size=args.batch_size,
                    include_mass=True,
                    num_workers=0,
                    pin_memory=False,
                )
            except Exception as e:
                print(f"  ! loader build failed: {e!r} (npz: {npz_group})")
                continue

            loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

            try:
                model = build_model(ckpt_path=ckpt_path, model_type=model_type,
                                    device=device)
            except Exception as e:
                print(f"  ! build failed: {e!r}")
                continue

            m = evaluate(model, model_type, loader, args.split, device,
                         K=args.rollout_K, rollout_batches=args.rollout_batches)
            print(f"  mse={m.mse:.3e}  |ΔE/E₀|={m.energy:.3e}  "
                  f"latency={m.latency_ms:.2f} ms/batch  rollout={m.rollout:.3e}")
            metrics_list.append(m)

    if not metrics_list:
        print("\n[eval] no successful runs; every checkpoint failed to load "
              "or had no matching dataset.")
        raise SystemExit(1)

    print("\n" + format_metrics_table(metrics_list))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in metrics_list], f, indent=2)
        print(f"[json] -> {args.json}")

    make_plots(metrics_list, args.out)


if __name__ == "__main__":
    main()
