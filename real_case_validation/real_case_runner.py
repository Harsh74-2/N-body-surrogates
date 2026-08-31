"""
real_case_runner.py
===================
Orchestration entry point for the real-case validation pipeline.

For every requested preset, this script:
  1. loads the IC (preset or user JSON) and rescales it into N-body units;
  2. runs a high-precision leapfrog reference (dt = coarse_dt / 100);
  3. for each trained checkpoint, runs an autoregressive rollout
     matching the surrogate's W=5 input contract;
  4. computes per-preset metrics in dimensionless units;
  5. writes per-preset plots + summary JSON;
  6. after all presets, writes a cross-preset dashboard and a markdown
     summary.

The CLI mirrors `evaluate_models.py` so the same `--ckpt` flags work
for both.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Make the script runnable both as `python -m real_case_validation.runner`
# (the canonical form) and as `python real_case_validation/real_case_runner.py`.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "real_case_validation"

from utils import configure_utf8_stdout, load_sibling_module, pick_device

configure_utf8_stdout()

from . import ic_loader
from . import unit_rescale
from . import references
from . import metrics
from . import plots as rv_plots
from . import kepler_check
from . import calibration

from pipeline_config import DAY_S, FEATURE_DIM, ModelType, REAL_CASE_WARMUP_WINDOW


# ── Constants ────────────────────────────────────────────────────────────────
YEAR_S = 365.25 * DAY_S
WINDOW_SIZE = REAL_CASE_WARMUP_WINDOW  # the W the surrogates were trained with


# Load `losses.py` for the per-model energy helper.
_loss_mod = load_sibling_module("nbody_losses", "losses.py")
total_energy_torch = _loss_mod.total_energy

# Load `stability_benchmark.py` for the shared no-grad eval rollout
# (`rollout_sliding`); the in-script `_rollout_sliding` below is a thin
# alias for it. The training-time rollout lives in `losses.rollout_energy_loss`
# and is intentionally kept separate (it builds a graph for BPTT).
_stab_mod = load_sibling_module("nbody_stability_benchmark", "stability_benchmark.py")
_rollout_sliding_impl = _stab_mod.rollout_sliding

# Load the model classes by file (matches evaluate_models.py).
_mlp_mod  = load_sibling_module("_rcv_mlp",  "mlp_train.py")
_lstm_mod = load_sibling_module("_rcv_lstm", "lstm_train.py")
_gnn_mod  = load_sibling_module("_rcv_gnn",  "gnn_train.py")
MLPSurrogate  = _mlp_mod.MLPSurrogate
LSTMSurrogate = _lstm_mod.LSTMSurrogate
GNNSurrogate  = _gnn_mod.GNNSurrogate


# ── Checkpoint loading (mirrors evaluate_models.build_model) ─────────────────
@dataclass
class LoadedModel:
    name:        str            # short, distinct display name
    model_type:  str
    model:       torch.nn.Module
    n_params:    int
    hidden:      int
    num_layers:  int
    num_passes:  int


def load_model(ckpt_path: str, model_type: str, device: torch.device,
               display_name: str | None = None) -> LoadedModel:
    """Load a trained checkpoint into a CPU/GPU `*Surrogate`.

    `display_name` overrides the label used to key this model in the
    report/plots. By default the model_type is used, which means two
    same-type checkpoints (e.g. single-step vs stable, or two different
    training N) would collide on the same key. Pass a distinct
    `display_name` per checkpoint to keep them separate.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {}) or {}
    hidden     = ckpt.get("hidden",     cfg_dict.get("hidden",     128))
    depth      = ckpt.get("depth",      cfg_dict.get("depth",      4))
    num_layers = ckpt.get("num_layers", cfg_dict.get("num_layers", 2))
    num_passes = ckpt.get("num_passes", cfg_dict.get("num_passes", 2))

    if model_type == ModelType.MLP:
        window_size = ckpt.get("window_size",
                               cfg_dict.get("window_size", REAL_CASE_WARMUP_WINDOW))
        in_features = ckpt.get("in_features",
                               cfg_dict.get("in_features", FEATURE_DIM))
        model = MLPSurrogate(window_size=window_size,
                             in_features=in_features,
                             hidden=hidden,
                             depth=depth)
    elif model_type == ModelType.LSTM:
        window_size = ckpt.get("window_size",
                               cfg_dict.get("window_size", REAL_CASE_WARMUP_WINDOW))
        in_features = ckpt.get("in_features",
                               cfg_dict.get("in_features", FEATURE_DIM))
        model = LSTMSurrogate(window_size=window_size,
                              in_features=in_features,
                              hidden=hidden,
                              num_layers=num_layers)
    elif model_type == ModelType.GNN:
        in_features = ckpt.get("in_features",
                               cfg_dict.get("in_features", FEATURE_DIM))
        model = GNNSurrogate(in_features=in_features,
                             hidden=hidden,
                             num_passes=num_passes)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device).eval()
    # Default the display name to the model_type so different architectures
    # show up as separate rows even when the ckpt filename collides
    # (`model_best.pt` for all three). `display_name` overrides it so
    # multiple same-type checkpoints can coexist in one report.
    return LoadedModel(
        name=(display_name or model_type.upper()),
        model_type=model_type,
        model=model,
        n_params=sum(p.numel() for p in model.parameters()),
        hidden=hidden,
        num_layers=num_layers,
        num_passes=num_passes,
    )


# ── Rollout helpers (per model type) ────────────────────────────────────────
def _rollout_sliding(model: torch.nn.Module,
                     window0: torch.Tensor,        # (W, N, F)
                     mass: torch.Tensor,           # (1, N)  (B, N) form)
                     n_steps: int) -> np.ndarray:
    """
    Thin alias for `stability_benchmark.rollout_sliding`; kept here
    for back-compat with the many call sites in this module.

    Sliding-window autoregressive rollout for every surrogate
    (MLP / LSTM / GNN). All three models were trained on real W=5
    windows and their `forward` takes `(B, W, N, F)` plus a `(B, N)`
    mass tensor. Routing the GNN through its full W-window forward
    (with message passing) is essential: an earlier version called
    `model.step`, which wraps the state as a W=1 window, and the
    GNN's `forward` only runs message passing for `t in range(1, W)`
    -- so W=1 ran *zero* message-passing rounds. Every OOD GNN
    number (including the large N=50 outlier) was produced on that
    degenerate no-message-passing path and was an artefact.

    The training-time rollout lives in `losses.rollout_energy_loss`
    and is intentionally kept separate (it builds a graph for BPTT).
    """
    return _rollout_sliding_impl(model, window0, mass, n_steps)


# ── Single-step evaluation (no rollout compounding) ─────────────────────────
def _single_step_predictions(model: torch.nn.Module,
                             ref_state: np.ndarray,    # (n_samples, N, 6)
                             mass: torch.Tensor,        # (1, N)
                             window_size: int) -> np.ndarray:
    """
    For every starting index `k` in `[window_size, n_samples - 1]`, build
    a warm-up window from `ref_state[k - W : k]` and ask the surrogate to
    predict the next frame. Returns a `(n_samples - W, N, 6)` array of
    independent single-step predictions — *no* autoregressive feedback
    into the next step's window.

    The array is roughly aligned to `ref_state[window_size:]` (the
    ground-truth "next frame" for every prediction); the per-index
    comparison is direct.

    This isolates the *bare* surrogate prediction error: the 1-3% MSE
    that the surrogates were trained on. The rollout-averaged error in
    `run_preset` is much larger because errors compound over the
    autoregressive loop.
    """
    n_samples = ref_state.shape[0]
    n_skip = n_samples - window_size
    if n_skip < 1:
        raise ValueError(
            f"need at least {window_size + 1} samples "
            f"({window_size} warm-up + 1 prediction), got {n_samples}.")
    preds = np.empty((n_skip, ref_state.shape[1], ref_state.shape[2]),
                     dtype=np.float32)
    with torch.no_grad():
        for k in range(n_skip):
            window = ref_state[k:k + window_size]            # (W, N, 6)
            window_t = torch.as_tensor(window, dtype=torch.float32) \
                              .unsqueeze(0).to(mass.device)  # (1, W, N, 6)
            pred = model(window_t, mass)                      # (1, N, 6)
            preds[k] = pred[0].cpu().numpy()
    return preds


def run_single_step(preset_spec: str,
                    loaded_models: list[LoadedModel],
                    device: torch.device,
                    out_dir: str,
                    quick: bool = False,
                    warmup: int | None = None,
                    dump_preds: bool = False) -> dict:
    """
    Single-step variant of `run_preset`. For every starting index the
    surrogate predicts the next frame **once** (no autoregressive
    feedback); the rollout error never compounds. The headline metric
    is the 1-3% single-step MSE the surrogates were trained on.

    Writes:
      - `<out_dir>/preset_<name>/ss_summary.json` with per-model metrics
      - one Markdown summary at `<out_dir>/single_step_report.md`
    """
    t0 = time.perf_counter()

    if Path(preset_spec).is_file():
        ric = ic_loader.load_custom(preset_spec)
    else:
        ric = ic_loader.load_preset(preset_spec)
    n_total = int(ric.mass.shape[0])
    n_samples = int(round(ric.duration_years * ric.sample_per_year))
    # `--warmup N` overrides the training-time W; otherwise we use the
    # constant from pipeline_config. Cap at n_samples // 4 so a long
    # warmup doesn't eat into the prediction budget.
    W = int(warmup) if (warmup is not None and warmup > 0) else WINDOW_SIZE
    W = min(W, max(5, n_samples // 4))
    # Clamp to the smallest model window_size — MLP/LSTM hard-reject
    # any W they weren't trained on (see the matching block in
    # `run_preset`).
    if loaded_models:
        model_Ws = [getattr(lm.model, "window_size", WINDOW_SIZE)
                    for lm in loaded_models]
        min_model_W = min(model_Ws) if model_Ws else WINDOW_SIZE
        if W != min_model_W:
            print(f"  [warmup] requested W={W} exceeds min model "
                  f"window_size={min_model_W}; clamping.")
            W = min_model_W
    if n_samples <= W:
        raise ValueError(
            f"preset {ric.name}: need at least {W + 1} samples "
            f"({W} warm-up + 1 prediction), got {n_samples}.")

    if ric.in_distribution:
        dt_N = float(ric.duration_years) / n_samples
    else:
        total_time_s = ric.duration_years * YEAR_S
        total_time_N = total_time_s / ric.scale.T
        dt_N = total_time_N / n_samples

    print(f"\n[single-step] {ric.name}")
    print(f"  N = {n_total}  samples = {n_samples}  "
          f"dt_N = {dt_N:.4e}  (N-body units)")

    # Reference trajectory once.
    if ric.reference == "kepler":
        ref_traj = _run_kepler_reference(ric, n_samples, dt_N)
    else:
        ref_traj = references.reference_leapfrog(
            ric.pos, ric.vel, ric.mass,
            dt_N=dt_N, n_steps=n_samples,
            epsilon=_prescaled_eps(ric),
            ref_substeps=100 if not quick else 20,
        )
    ref_state = np.concatenate([ref_traj["pos"], ref_traj["vel"]], axis=-1)
    mass_t = torch.as_tensor(ric.mass, dtype=torch.float32,
                             device=device).unsqueeze(0)

    # ── Per-model single-step predictions ──────────────────────────────
    per_model: dict = {}
    pred_stack: dict[str, np.ndarray] = {}   # name -> (n_skip, N, 6)
    char_L = ric.characteristic_length_m / ric.scale.L           # = 1.0
    n_skip = n_samples - W
    for lm in loaded_models:
        print(f"  [single-step] {lm.name}  ({lm.model_type}, "
              f"{lm.n_params:,} params)")
        preds = _single_step_predictions(lm.model, ref_state,
                                        mass_t, W)
        # Direct element-wise comparison against the corresponding
        # reference frame; no autoregressive feedback in the window.
        target = ref_state[W:]                         # (n_skip, N, 6)
        # Position slices only — this is the dimensionally-meaningful
        # indicator of how close the surrogate is to the leapfrog step.
        per_body = np.linalg.norm(preds[..., :3] - target[..., :3],
                                  axis=-1)                       # (n_skip, N)
        # Normalise by L: per_body / char_L == per_body since char_L=1.
        mean_err_L = float(per_body.mean())
        max_err_L  = float(per_body.max())
        mse_pos    = float(np.mean((preds[..., :3] - target[..., :3]) ** 2))
        mse_state  = float(np.mean((preds - target) ** 2))
        # Energy drift on the *predicted* state vs the *target* state.
        # Single-step so the predicted energy is the only one that
        # matters; the target's energy is the leapfrog reference.
        eps = _prescaled_eps(ric)
        E_pred = np.empty(n_skip)
        E_tgt  = np.empty(n_skip)
        mass_cpu = mass_t.detach().cpu()
        for k in range(n_skip):
            E_pred[k] = total_energy_torch(
                torch.as_tensor(preds[k, :, :3], dtype=torch.float32),
                torch.as_tensor(preds[k, :, 3:6], dtype=torch.float32),
                mass_cpu, eps=eps, g=1.0).item()
            E_tgt[k] = total_energy_torch(
                torch.as_tensor(target[k, :, :3], dtype=torch.float32),
                torch.as_tensor(target[k, :, 3:6], dtype=torch.float32),
                mass_cpu, eps=eps, g=1.0).item()
        energy_drift = float(np.abs(E_pred - E_tgt).mean()
                             / max(abs(E_tgt[0]), 1e-8))
        energy_drift_max = float(
            np.abs(E_pred - E_tgt).max() / max(abs(E_tgt[0]), 1e-8))
        per_model[lm.name] = {
            "mse_position":        mse_pos,
            "mse_state":           mse_state,
            "mean_error_over_L":   mean_err_L,
            "max_error_over_L":    max_err_L,
            "mean_err_pct":        100.0 * mean_err_L,
            "max_err_pct":         100.0 * max_err_L,
            "energy_drift":        energy_drift,
            "energy_drift_max":    energy_drift_max,
            "n_predictions":       n_skip,
        }
        print(f"    ✓ {n_skip} single-step predictions  "
              f"mean_err = {mean_err_L:.3e} L  "
              f"= {100.0 * mean_err_L:.3f} %")
        if dump_preds:
            pred_stack[lm.name] = preds
        else:
            del preds

    payload = {
        "name":             ric.name,
        "label":            ric.label,
        "in_distribution":  ric.in_distribution,
        "n_bodies":         n_total,
        "n_samples":        n_samples,
        "dt_N":             dt_N,
        "duration_years":   ric.duration_years,
        "n_predictions":    n_skip,
        "per_model":        per_model,
        "wallclock_s":      float(time.perf_counter() - t0),
    }

    preset_dir = Path(out_dir) / f"preset_{ric.name}"
    preset_dir.mkdir(parents=True, exist_ok=True)
    with open(preset_dir / "ss_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, "item") else str(o))
    print(f"  → {preset_dir}/ss_summary.json  "
          f"(elapsed {payload['wallclock_s']:.1f}s)")

    # Optional: persist the single-step arrays to disk so external tools
    # (notably `build_interactive_animations.py`) can rebuild trajectory
    # visualisations without re-running the surrogates. Files:
    #   preset_dir/preds.npy       — (n_models, n_skip, N, 6) float32
    #   preset_dir/ref_pos.npy     — (n_skip, N, 3) float64, the
    #                                reference positions the windows were
    #                                built from (= ref_state[W:])
    #   preset_dir/preds_meta.json — model list + frame count + dt_N + W
    # Because every prediction was compared against exactly
    # `ref_state[W:]`, `ref_pos.npy` is the same slice the errors in
    # `ss_summary.json` were computed from — dump and summary agree by
    # construction.
    if dump_preds:
        models_dumped = list(pred_stack.keys())
        preds_stack = np.stack(
            [pred_stack[m] for m in models_dumped], axis=0)
        np.save(preset_dir / "preds.npy",
                preds_stack.astype(np.float32))
        np.save(preset_dir / "ref_pos.npy",
                ref_state[W:, :, :3].astype(np.float64))
        scale = ric.scale.to_dict()
        meta = {
            "preset":         ric.name,
            "n_bodies":       n_total,
            "n_predictions":  int(n_skip),
            "models":         models_dumped,
            "dt_N":           dt_N,
            "W":              W,
            "reference":      ric.reference,
            "L_m":            scale["L_m"],
            "T_s":            scale["T_s"],
        }
        with open(preset_dir / "preds_meta.json", "w",
                  encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        print(f"  → {preset_dir}/preds.npy {preds_stack.shape}  "
              f"+ ref_pos.npy + preds_meta.json")

    return payload


def write_single_step_markdown(payloads: list[dict], out_path: str) -> None:
    """Markdown report for the single-step real-life validation."""
    lines = []
    lines.append("# Real-Case Validation, Single-Step Report\n")
    lines.append(
        "Same six surrogate variants (MLP / LSTM / GNN × single + "
        "stable) evaluated on real Solar-System initial conditions, "
        "but **without autoregressive rollout**. Each surrogate is "
        "asked to predict the *next frame only* from a warm-up window "
        "of `WINDOW_SIZE` leapfrog frames; the prediction is then "
        "compared directly against the leapfrog reference at that "
        "next frame. Errors do not compound because the window is "
        "always re-built from the reference, never from the model's "
        "own output.\n")
    lines.append("This is the *bare* prediction error — the 1-3 % "
                 "single-step MSE the surrogates were trained on. "
                 "Compare with the rollout-averaged report "
                 "(`real_case_report.md` in the same parent "
                 "directory) to see how much the error compounds "
                 "after autoregressive feedback.\n")
    lines.append("## Per-preset single-step error %\n")
    MODEL_ORDER = ("MLP", "MLP_stable", "LSTM", "LSTM_stable",
                   "GNN", "GNN_stable")
    for p in payloads:
        tag = "in-distribution baseline" if p.get("in_distribution") else "out-of-distribution"
        lines.append(f"### `{p['name']}` — {p.get('label', '')}")
        lines.append(f"- bodies: {p['n_bodies']}, samples: {p['n_samples']}, "
                     f"predictions: {p['n_predictions']}, "
                     f"dt_N = {p['dt_N']:.3e} ({tag})")
        lines.append("")
        lines.append("| model | MSE pos | mean err % | max err % | "
                     "energy drift |")
        lines.append("|---|---|---|---|---|")
        for m in MODEL_ORDER:
            mm = p["per_model"].get(m)
            if mm is None:
                continue
            lines.append(
                f"| {m} | {mm['mse_position']:.3e} | "
                f"{mm['mean_err_pct']:.2f} % | "
                f"{mm['max_err_pct']:.2f} % | "
                f"{mm['energy_drift']:.3e} |"
            )
        lines.append("")

    # Cross-preset aggregate.
    lines.append("## Cross-preset aggregate (single-step mean error %)\n")
    lines.append("Mean of `mean_err_%` across the presets that ran:\n")
    in_dist = [p for p in payloads if p.get("in_distribution")]
    ood     = [p for p in payloads if not p.get("in_distribution")]
    lines.append("| model | in-distribution | Solar-System OOD |")
    lines.append("|---|---|---|")
    in_means = {m: [] for m in MODEL_ORDER}
    ood_means = {m: [] for m in MODEL_ORDER}
    for p in payloads:
        for m in MODEL_ORDER:
            mm = p["per_model"].get(m)
            if mm is None:
                continue
            (in_means if p.get("in_distribution") else ood_means)[m].append(
                mm["mean_err_pct"])
    for m in MODEL_ORDER:
        if in_means[m] or ood_means[m]:
            in_avg = (sum(in_means[m]) / len(in_means[m])
                      if in_means[m] else float("nan"))
            ood_avg = (sum(ood_means[m]) / len(ood_means[m])
                       if ood_means[m] else float("nan"))
            lines.append(f"| {m} | {in_avg:.2f} % | {ood_avg:.2f} % |")
    lines.append("")
    if in_dist and ood:
        lines.append(
            "The single-step MSE is the *honest* headline number — "
            "the rollout-averaged error in the autoregressive report "
            "grows large because errors compound over the loop. The "
            "single-step number is the one to cite in the "
            "abstract.\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Single-preset run ───────────────────────────────────────────────────────
def run_preset(preset_spec,                    # name | path-to-JSON
               loaded_models: list[LoadedModel],
               device: torch.device,
               out_dir: str,
               quick: bool = False,
               warmup: int | None = None,
               ensemble: list[str] | None = None,
               dump_preds: bool = False) -> dict:
    """
    Run the full pipeline for one preset and return a payload dict
    ready to write into `real_case_report.json`.

    Parameters
    ----------
    warmup     : override the inference warm-up window (default: use
                 the training-time `WINDOW_SIZE` from pipeline_config).
    ensemble   : list of model display names (e.g. ["LSTM", "GNN"]) to
                 ensemble at inference — per-component mean of the
                 predictions; the ensemble is appended to summary.json
                 as `ensemble_<sorted_names>`.
    dump_preds : persist `preds.npy` + `book_pos.npy` + `preds_meta.json`
                 into the preset directory so external tools can rebuild
                 visualisations without re-running the surrogates.
    """
    t0 = time.perf_counter()

    # ── Load + rescale IC ───────────────────────────────────────────────
    if Path(preset_spec).is_file():
        ric = ic_loader.load_custom(preset_spec)
    else:
        ric = ic_loader.load_preset(preset_spec)
    n_total   = int(ric.mass.shape[0])
    n_samples = int(round(ric.duration_years * ric.sample_per_year))
    # `--warmup N` overrides the training-time W; otherwise we use the
    # constant from pipeline_config. Cap at n_samples // 4 so a long
    # warmup doesn't eat into the rollout budget. Also clamp to the
    # smallest model window_size — MLP/LSTM were trained at fixed W and
    # their `forward()` hard-rejects any other W; widening past the
    # training-time W would silently break them. The clamp is the
    # honest way to express "we can't go past the trained W without
    # retraining", which is the no-retrain constraint.
    W = int(warmup) if (warmup is not None and warmup > 0) else WINDOW_SIZE
    W = min(W, max(5, n_samples // 4))
    if loaded_models:
        model_Ws = [getattr(lm.model, "window_size", WINDOW_SIZE)
                    for lm in loaded_models]
        min_model_W = min(model_Ws) if model_Ws else WINDOW_SIZE
        if W != min_model_W:
            print(f"  [warmup] requested W={W} exceeds min model "
                  f"window_size={min_model_W}; clamping.")
            W = min_model_W
    # The reference records n_samples frames (indices 0..n_samples-1).
    # We use the first WINDOW_SIZE for the warm-up, then run the
    # surrogate for (n_samples - WINDOW_SIZE) *new* steps starting
    # from frame WINDOW_SIZE. So the surrogate rollout yields
    # n_samples - WINDOW_SIZE + 1 frames (initial + new); the
    # comparison slice from the reference has the same length.
    n_rollout = n_samples - W
    if n_rollout < 1:
        raise ValueError(
            f"preset {ric.name}: need at least {W+1} samples "
            f"({W} warm-up + 1 rollout), got {n_samples}. "
            f"Increase duration_years or sample_per_year.")

    # Coarse dt (in N-body time units), the time interval between
    # *recorded* samples; this is what the surrogate is asked to
    # predict one step at a time.
    if ric.in_distribution:
        # In the synthetic disc, "duration_years" is actually in
        # crossing-time units; 1 year here = 1 N-body time unit.
        dt_N = float(ric.duration_years) / n_samples
    else:
        total_time_s = ric.duration_years * YEAR_S
        total_time_N = total_time_s / ric.scale.T
        dt_N = total_time_N / n_samples

    print(f"\n[preset] {ric.name}")
    print(f"  N = {n_total}  samples = {n_samples}  "
          f"dt_N = {dt_N:.4e}  (N-body units)")
    print(f"  scale: M = {ric.scale.M:.4e} kg, "
          f"L = {ric.scale.L:.4e} m, T = {ric.scale.T:.4e} s")

    # ── Reference trajectory ────────────────────────────────────────────
    # Sub-stepping keeps the reference honest even when coarse_dt is
    # surprisingly large after a rescaling to a big L.
    if ric.reference == "kepler":
        # Only the Sun-Earth 2-body preset uses this. Pull a, e from
        # the body list (Earth at a_au, e=0.0167).
        ref_traj = _run_kepler_reference(ric, n_samples, dt_N)
    else:
        ref_traj = references.reference_leapfrog(
            ric.pos, ric.vel, ric.mass,
            dt_N=dt_N, n_steps=n_samples,
            epsilon=_prescaled_eps(ric),
            ref_substeps=100 if not quick else 20,
        )

    # Warm-up window of W frames (the first W entries of the reference).
    # Build the *state* tensor (x, y, z, vx, vy, vz), concat pos and
    # vel along the last axis to give the (W, N, 6) layout the
    # surrogates were trained on.
    ref_state = np.concatenate(
        [ref_traj["pos"][:W], ref_traj["vel"][:W]],
        axis=-1,
    )                                                          # (W, N, 6)
    warm_window = torch.as_tensor(ref_state, dtype=torch.float32, device=device)
    # Every surrogate's forward expects mass of shape (B, N) (the GNN
    # also accepts (N,) but the LSTM does not). Use the (B=1, N) form
    # uniformly to keep the sliding-window rollout identical across models.
    mass_t      = torch.as_tensor(ric.mass, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N)

    # ── Per-surrogate rollouts ──────────────────────────────────────────
    surrogate_trajs: dict[str, np.ndarray] = {}
    surrogate_energies: dict[str, np.ndarray] = {}
    for lm in loaded_models:
        print(f"  [rollout] {lm.name}  ({lm.model_type}, {lm.n_params:,} params)")
        # All three models are rolled out with the in-distribution
        # sliding W-window (see _rollout_sliding). The GNN must go
        # through its full W-window forward so message passing runs;
        # `model.step` (W=1) skips it.
        traj = _rollout_sliding(lm.model, warm_window, mass_t, n_rollout)
        # Energy series in N-body units. Surrogate trajectory is
        # (T, N, 6) = pos || vel, so velocities come straight from
        # the model's predictions, no finite-difference approximation
        # needed (the model was trained to predict velocities too).
        vel = traj[..., 3:6]
        E = np.empty(traj.shape[0])
        # traj is numpy on CPU (the rollout returns .cpu().numpy()), so the
        # pos/vel tensors below are CPU. Pull mass onto CPU too so the three
        # tensors share a device inside total_energy_torch (it rejects mixed
        # cuda/cpu, e.g. when the model ran on GPU). The result is a scalar
        # via .item(), so computing on CPU costs nothing here.
        mass_cpu = mass_t.detach().cpu()
        for k in range(traj.shape[0]):
            E[k] = total_energy_torch(
                torch.as_tensor(traj[k, :, :3], dtype=torch.float32),
                torch.as_tensor(vel[k], dtype=torch.float32),
                mass_cpu, eps=_prescaled_eps(ric), g=1.0,
            ).item()
        surrogate_trajs[lm.name]    = traj
        surrogate_energies[lm.name] = E
        print(f"    ✓ {traj.shape[0]} frames  "
              f"max drift = {metrics.energy_drift_normalised(E).max():.3e}")

    # ── Metrics ─────────────────────────────────────────────────────────
    char_L = ric.characteristic_length_m / ric.scale.L   # = 1.0 by construction
    per_model_metrics: dict = {}
    # ── Align surrogate trajectories to the reference ───────────────
    # The surrogate's first frame is its warm-up anchor (surrogate
    # index 0 = ref frame WINDOW_SIZE - 1, since warm_window is
    # ref[:WINDOW_SIZE]). The surrogate's predictions start at
    # *its* index 1 and correspond to ref frames WINDOW_SIZE,
    # WINDOW_SIZE + 1, ... The reference has `n_samples - WINDOW_SIZE`
    # predictions-after-warm-up available (frames WINDOW_SIZE ..
    # n_samples - 1). So:
    #   * drop surrogate index 0 (its warm-up anchor);
    #   * keep the next `n_samples - WINDOW_SIZE` frames;
    #   * compare against the reference slice starting at WINDOW_SIZE.
    # Before this fix, the slice was off-by-one in either direction
    # (depending on which rounding the previous code chose), so the
    # surrogate-vs-reference error metric was contaminated by a 1-step
    # offset on every frame. The shape-mismatch error surfaced once
    # the rollout path was corrected for the new (longer) surrogate
    # trajectory after the rollout-path bugfix in `losses.py`.
    n_rollout_ref = ref_traj["pos"].shape[0] - W
    for model_name in list(surrogate_trajs.keys()):
        traj = surrogate_trajs[model_name]
        # The surrogate predicts (n_rollout) new frames after its
        # warm-up anchor. Trim index 0 (anchor) and any extra
        # tail frames beyond the reference's prediction range.
        traj = traj[1:1 + n_rollout_ref]
        surrogate_trajs[model_name]    = traj
        surrogate_energies[model_name] = surrogate_energies[model_name][1:1 + n_rollout_ref]

    # ── Optional inference ensemble (no retraining) ───────────────────
    # If the user passed `--ensemble lstm,gnn`, take the per-component
    # mean of the trimmed surrogate trajectories and inject it as a
    # synthetic row. The ensemble name is deterministic so the same
    # set always produces the same key in summary.json.
    if ensemble:
        present = [m for m in ensemble if m in surrogate_trajs]
        if len(present) < 2:
            print(f"  [ensemble] only {present} in loaded models; "
                  f"need ≥2. Skipping ensemble.")
        else:
            key = "ensemble_" + "_".join(present)
            ens_traj = np.mean(
                np.stack([surrogate_trajs[m] for m in present], axis=0),
                axis=0)
            # For energies we don't have per-component surrogate energy
            # arrays of matching shape; report the mean drift of the
            # constituent surrogates so the ensemble row has an energy
            # number. The plots only show the position slice, so the
            # energy drift is informational.
            ens_energy = np.mean(
                np.stack([surrogate_energies[m] for m in present], axis=0),
                axis=0)
            surrogate_trajs[key]    = ens_traj.astype(np.float32)
            surrogate_energies[key] = ens_energy.astype(np.float64)
            print(f"  [ensemble] {key} = mean({', '.join(present)})")
    for model_name, traj in surrogate_trajs.items():
        # Surrogate trajectory is (T, N, 6) = pos || vel. The reference's
        # `pos` and `vel` are (T, N, 3); concat to get the same 6-feature
        # layout for the comparison. The surrogate was already truncated
        # above to the reference's `n_rollout` length, so the slice
        # below is guaranteed to be in-bounds.
        n_traj = traj.shape[0]
        ref_state_rollout = np.concatenate(
            [ref_traj["pos"][W:W + n_traj],
             ref_traj["vel"][W:W + n_traj]],
            axis=-1,
        )                                                  # (T, N, 6)
        if traj.shape != ref_state_rollout.shape:
            raise RuntimeError(
                f"shape mismatch for {model_name}: "
                f"surrogate {traj.shape}, reference {ref_state_rollout.shape}")
        # Compare only the *position* slice, that's the project metric
        # and the only one with a well-defined physical interpretation
        # in dimensionless units.
        terr = metrics.trajectory_error_normalised(
            traj[..., :3], ref_state_rollout[..., :3], char_L)
        per_model_metrics[model_name] = {
            "mse_state":             metrics.trajectory_mse(
                traj, ref_state_rollout),
            "mse_position":          metrics.trajectory_mse(
                traj[..., :3], ref_state_rollout[..., :3]),
            "max_error_over_L":      terr["max_error_over_L"],
            "mean_error_over_L":     terr["mean_error_over_L"],
            "frames_before_half_L":  terr["frames_before_threshold"],
            "max_energy_drift":      float(
                metrics.energy_drift_normalised(
                    surrogate_energies[model_name]).max()),
        }

    # ── Per-preset artefacts ────────────────────────────────────────────
    preset_dir = Path(out_dir) / f"preset_{ric.name}"
    preset_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory plot: plot only the position slice (last axis is
    # pos || vel, but xy plots only need (T, N, 3)). Reference slice
    # is aligned to the surrogate predictions (frames WINDOW_SIZE ..
    # WINDOW_SIZE + n_rollout_ref) so the error-vs-reference subplot
    # can subtract them without a shape mismatch.
    ref_pos_slice    = ref_traj["pos"][W:W + n_rollout_ref]
    ref_energy_slice = ref_traj["energy"][W:W + n_rollout_ref]
    rv_plots.plot_trajectory_per_model(
        preset_name=ric.name,
        ref_traj=ref_pos_slice,
        surrogate_trajs={k: v[..., :3] for k, v in surrogate_trajs.items()},
        names=ric.names,
        out_dir=str(preset_dir),
    )
    rv_plots.plot_energy(
        preset_name=ric.name,
        ref_energy=ref_energy_slice,
        surrogate_energies=surrogate_energies,
        out_path=str(preset_dir / "energy.png"),
    )
    rv_plots.plot_error_vs_reference(
        preset_name=ric.name,
        ref_traj=ref_pos_slice,
        surrogate_trajs={k: v[..., :3] for k, v in surrogate_trajs.items()},
        out_path=str(preset_dir / "error_vs_reference.png"),
    )

    # ── Per-small-body vs closed-form Kepler ("book") reference ────────
    # For every non-primary body we propagate an *independent* 2-body
    # closed-form Kepler orbit using the body's osculating (a, e) from
    # its ICs and the Sun's mass (treating Sun+body as isolated). This
    # gives the supervisor a "book" prediction of where each small
    # body should be, ignoring perturbations from other planets —
    # exactly the trajectory an introductory astronomy textbook would
    # compute for that body. The leapfrog reference differs from the
    # book prediction only by the gravitational pull of the other
    # planets; the surrogate differs from the leapfrog by its learned
    # error. The radial-error panel makes all three separations
    # readable on the same plot.
    if not ric.in_distribution:
        # Route each body's book-orbit primary: planets -> Sun,
        # Galileans -> Jupiter (auto-detected by body name).
        primary_for_body = _default_primary_for_body(ric, primary_idx=0)
        book_pos = _compute_book_orbits(ric, n_rollout_ref, dt_N,
                                        primary_idx=0,
                                        primary_for_body=primary_for_body)

        # Per-body linear drift calibration (post-hoc, no retraining).
        # Fit r_predicted = a · r_book + b on the first 25 % of frames
        # of the surrogate rollout, apply to the remaining 75 %, and
        # report the standard metrics alongside the un-calibrated ones.
        # `per_model_metrics` is built ABOVE (line ~628) so we read from
        # `surrogate_trajs` directly here.
        for model_name, traj in surrogate_trajs.items():
            try:
                cals = calibration.fit_per_body(
                    traj, book_pos, primary_idx=0,
                    primary_for_body=primary_for_body, cal_frac=0.25)
                cal_metrics = calibration.calibration_metrics(
                    traj, book_pos, cals,
                    primary_idx=0, primary_for_body=primary_for_body,
                    cal_frac=0.25, char_L=1.0)
                per_model_metrics[model_name]["calibrated"] = cal_metrics
            except Exception as e:
                # Calibration is best-effort — never block the report.
                per_model_metrics[model_name]["calibrated"] = {
                    "error": repr(e),
                }
        # Per-model plots: one figure per (preset, model_variant) so the
        # supervisor sees a clean 3-line plot (book + reference + this
        # one surrogate) rather than a 6-line hodge-podge. The runner
        # owns the filename convention: <base>_<MODEL>.png.
        for model_name, traj in surrogate_trajs.items():
            tag = str(model_name).replace(" ", "_")
            rv_plots.plot_small_bodies_vs_book(
                preset_name=ric.name,
                ric=ric,
                ref_pos_slice=ref_pos_slice,
                surrogate_traj=traj,
                model_name=model_name,
                book_trajs=book_pos,
                out_path=str(preset_dir / f"small_bodies_vs_book_{tag}.png"),
                primary_idx=0,
                primary_for_body=primary_for_body,
            )
            # "Predicted vs book" 2D scatter: the body's predicted
            # position is plotted against its closed-form Kepler
            # position (in the body's primary frame). A perfect model
            # sits on the y = x diagonal. This is the supervisor's
            # most direct read-off of how close this surrogate is to
            # the 2-body analytical answer, separate from how close it
            # is to the leapfrog. Per-model so the user can read off
            # one number per body per variant.
            rv_plots.plot_predicted_vs_book(
                preset_name=ric.name,
                ric=ric,
                ref_pos_slice=ref_pos_slice,
                surrogate_traj=traj,
                model_name=model_name,
                book_trajs=book_pos,
                out_path=str(preset_dir / f"predicted_vs_book_{tag}.png"),
                primary_idx=0,
                primary_for_body=primary_for_body,
            )

    payload = {
        "name":             ric.name,
        "label":            ric.label,
        "in_distribution":  ric.in_distribution,
        "n_bodies":         n_total,
        "n_samples":        n_samples,
        "dt_N":             dt_N,
        "duration_years":   ric.duration_years,
        "scale":            ric.scale.to_dict(),
        "per_model":        per_model_metrics,
        "wallclock_s":      float(time.perf_counter() - t0),
    }

    # ── Kepler's 3rd-law check (reference only) ────────────────────────
    # The reference integrator is deterministic and should preserve
    # T²/a³ to a few parts in 10⁴ over a 10-yr window for the inner
    # planets. This sanity-checks the *reference* (not the surrogates,
    # which fail the 3rd law on the first frame by design). Skipped
    # on the in-distribution disc baseline, which has no clear primary.
    if not ric.in_distribution:
        kepler_rows = kepler_check.kepler_table(
            ref_traj["pos"], ref_traj["vel"], ric.mass, dt_N,
            names=ric.names,
            scale_M_kg=ric.scale.M,
            scale_L_m=ric.scale.L,
            scale_T_s=ric.scale.T,
        )
        payload["kepler_check"] = kepler_rows
        n_kepler = sum(1 for r in kepler_rows
                       if not r["is_primary"]
                       and not math.isnan(r["deviation_pct"]))
        print(f"  [kepler] reference obeys T²/a³ for {n_kepler}/"
              f"{n_total - 1} bodies (NaN = orbit too long for window)")
    else:
        payload["kepler_check"] = []
        print(f"  [kepler] skipped (in-distribution baseline)")
    with open(preset_dir / "summary.json", "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, "item") else str(o))
    print(f"  → {preset_dir}/  (elapsed {payload['wallclock_s']:.1f}s)")

    # Also include the raw reference / surrogate arrays in the
    # dashboard payload (so the dashboard can be drawn after the
    # loop). We keep these *only in memory* and pass them back up
    # to the caller. The reference slice is aligned to the surrogate
    # predictions (frames WINDOW_SIZE : WINDOW_SIZE + n_rollout_ref) so
    # the per-step plots in `plot_dashboard` can do `traj - ref_pos`
    # without a broadcast error.
    payload["_ref_pos"]    = ref_traj["pos"][W:W + n_rollout_ref]
    payload["_ref_energy"] = ref_traj["energy"][W:W + n_rollout_ref]
    payload["_surrogate_trajs"]    = surrogate_trajs
    payload["_surrogate_energies"] = surrogate_energies

    # Optional: persist the rollout arrays to disk so external tools
    # (notably `make_animations.py`) can rebuild visualisations
    # without re-running the surrogates. Files:
    #   preset_dir/preds.npy      — (n_models, T, N, 6) float32
    #   preset_dir/book_pos.npy   — (T, N, 3) float64
    #   preset_dir/preds_meta.json — model list + frame count + dt_N
    if dump_preds:
        preds_stack = np.stack(
            [surrogate_trajs[m] for m in surrogate_trajs], axis=0)
        np.save(preset_dir / "preds.npy", preds_stack.astype(np.float32))
        # For in-distribution presets `book_pos` is never assigned (the
        # runner skips the closed-form Kepler reference). Fall back to
        # the leapfrog reference trajectory so the animation pipeline
        # always has a "book" to render against.
        #
        # `preds.npy` is the stacked surrogate trajectories AFTER the
        # W-strip at line 621, so its time axis has length
        # `n_rollout_ref = ref_traj["pos"].shape[0] - W`. We must dump
        # `book_pos` with the same length (slice off the warmup frames)
        # or the animation pipeline's `surr_full - book_pos` broadcast
        # at `make_animations.py:611` raises
        #     ValueError: operands could not be broadcast together
        #     with shapes (n_rollout_ref,N,3) (T,N,3)
        # The same `ref_pos_slice` is what the trajectory/error plots
        # use (`plot_trajectory_per_model(ref_traj=ref_pos_slice)` at
        # line 696), so reusing it keeps dump + plot + animation
        # perfectly aligned.
        book_for_dump = (book_pos if not ric.in_distribution
                         else ref_traj["pos"][W:W + n_rollout_ref])
        np.save(preset_dir / "book_pos.npy",
                book_for_dump.astype(np.float64))
        meta = {
            "preset": ric.name,
            "n_bodies": n_total,
            "n_rollout_ref": int(n_rollout_ref),
            "models": list(surrogate_trajs.keys()),
            "dt_N": dt_N,
            "W": W,
        }
        with open(preset_dir / "preds_meta.json", "w",
                  encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
    return payload


def _prescaled_eps(ric: ic_loader.RescaledIC) -> float:
    """
    Softening in dimensionless units.

    In-distribution discs use ε = 0.1 (training default). For real
    presets the natural ε in N-body units is the dimensionless
    counterpart of an "absurdly small" physical ε, so we just use
    1e-4 (tighter than training, keeps the reference clean).
    """
    return 0.1 if ric.in_distribution else 1e-4


def _run_kepler_reference(ric: ic_loader.RescaledIC, n_samples: int,
                          dt_N: float) -> dict:
    """
    Closed-form Kepler reference for the Sun-Earth 2-body preset.

    Only valid for N=2 with one body much heavier than the other.
    """
    # Index 0 is the Sun (primary), index 1 is Earth (secondary).
    M1, M2 = float(ric.mass[0]), float(ric.mass[1])
    mu = M1 + M2
    r_vec = ric.pos[1] - ric.pos[0]
    v_vec = ric.vel[1] - ric.vel[0]
    r0 = float(np.linalg.norm(r_vec))
    v0 = float(np.linalg.norm(v_vec))
    # Compute osculating a and e directly from the 2-body state so the
    # closed-form reference matches the preset's actual eccentricity.
    rv_dot = float(np.dot(r_vec, v_vec))
    e_vec = ((v0 ** 2 - mu / r0) * r_vec - rv_dot * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    a = 1.0 / (2.0 / r0 - v0 ** 2 / mu)
    if a <= 0:
        raise ValueError("non-positive semi-major axis in Kepler reference")
    t = np.arange(n_samples) * dt_N
    return references.reference_kepler(M1, M2, a, e, t, g=1.0)


def _default_primary_for_body(ric: ic_loader.RescaledIC,
                              primary_idx: int = 0) -> np.ndarray:
    """
    Build a length-N int array mapping every body index to its book-
    orbit primary. Defaults to `primary_idx` (Sun) for every body
    except the Sun itself; overrides a handful of moons to orbit
    their planet (Galileans -> Jupiter) so the 2-body "book"
    approximation makes physical sense for them.

    A moon whose computed primary isn't in the preset (shouldn't
    happen for the supported presets) falls back to the Sun.
    """
    n_total = int(ric.mass.shape[0])
    pfb = np.full(n_total, primary_idx, dtype=np.int64)

    # Try to detect Jupiter by name; if the preset has it, route
    # Galilean moons around Jupiter rather than the Sun. The
    # 2-body Sun+Moon approximation puts Io's book orbit at
    # ~5 AU, which is nonsense next to a real ~0.003 AU orbit.
    names = list(getattr(ric, "names", []) or [])

    # Detect Jupiter by name. Two naming conventions appear in the
    # built-in presets: "Jupiter" (planet row) plus plain "Io" /
    # "Europa" / "Ganymede" / "Callisto" (moon rows); OR a single
    # "Jupiter-Io" / "Jupiter-Europa" / etc. naming style. The latter
    # convention does not appear as a row "Jupiter" in the names
    # list, so we ALSO match any name ending in "-Io" / "-Europa" /
    # etc. The same idea handles "Saturn-Titan", "Pluto-Charon", etc.
    jup_idx = None
    if "Jupiter" in names:
        jup_idx = names.index("Jupiter")

    galilean_suffixes = ("-Io", "-Europa", "-Ganymede", "-Callisto",
                         " Io", " Europa", " Ganymede", " Callisto")
    for i, nm in enumerate(names):
        # Case 1: Galilean-prefixed style (e.g. "Jupiter-Io") — use
        # the prefix to find the host planet row.
        for suf in galilean_suffixes:
            if nm.endswith(suf):
                host = nm[:-len(suf)]
                if host in names:
                    pfb[i] = names.index(host)
                elif jup_idx is not None:
                    # Prefix names a planet we don't have a row
                    # for (rare) — fall back to Jupiter.
                    pfb[i] = jup_idx
                break
        else:
            # Case 2: plain moon names — bind to Jupiter if present.
            if nm in ("Io", "Europa", "Ganymede", "Callisto") \
                    and jup_idx is not None:
                pfb[i] = jup_idx

    return pfb


def _compute_book_orbits(ric: ic_loader.RescaledIC, n_steps: int,
                         dt_N: float, primary_idx: int = 0,
                         primary_for_body: np.ndarray | None = None,
                         ) -> np.ndarray:
    """
    Closed-form Kepler orbits for every body in `ric`. Each body
    uses its own primary (from `primary_for_body`, or Sun by
    default) — planets orbit the Sun, moons orbit their planet.

    This is the "book" reference: where each body would be if
    *only* its primary's gravity acted on it. Used by
    `plot_small_bodies_vs_book` to give the supervisor a visual
    comparison — leapfrog-vs-book shows the size of the
    perturbations, surrogate-vs-book shows the total prediction
    error.

    Returns
    -------
    book_pos : (n_steps, N, 3) — closed-form Kepler positions
        over `n_steps` at sample interval `dt_N`, in the same
        N-body units as `ric.pos`, aligned to the reference
        trajectory. Each primary is pinned at its IC (the μ-
        recursive solution would put it on a small counter-orbit
        about the COM, which would offset every plot).
    """
    n_total = int(ric.mass.shape[0])
    t = np.arange(n_steps) * dt_N
    book_pos = np.zeros((n_steps, n_total, 3), dtype=np.float64)

    if primary_for_body is None:
        primary_for_body = _default_primary_for_body(ric, primary_idx)

    # Pin each primary at its IC. We keep one pinned position per
    # primary index, since several bodies may share it (e.g. the
    # Galilean moons all use Jupiter).
    pinned_primaries = {}
    for body_i in range(n_total):
        pri = int(primary_for_body[body_i])
        if pri not in pinned_primaries:
            pinned_primaries[pri] = ric.pos[pri].copy()
        # Also pin the primary itself to its IC if body_i is its own
        # primary (self-reference shouldn't happen, but be safe).
        if pri == body_i:
            book_pos[:, body_i, :] = ric.pos[body_i]
            continue
        M_primary = float(ric.mass[pri])
        primary_pos_0 = pinned_primaries[pri]

        mu = M_primary + float(ric.mass[body_i])
        # Relative state at t=0, in the primary's frame.
        r_vec = ric.pos[body_i] - ric.pos[pri]
        v_vec = ric.vel[body_i] - ric.vel[pri]
        r0 = float(np.linalg.norm(r_vec))
        v0 = float(np.linalg.norm(v_vec))
        # Osculating orbit elements.
        rv_dot = float(np.dot(r_vec, v_vec))
        e_vec = ((v0 ** 2 - mu / r0) * r_vec - rv_dot * v_vec) / mu
        e = float(np.linalg.norm(e_vec))
        if e < 1e-12:
            e = 0.0
        a = 1.0 / (2.0 / r0 - v0 ** 2 / mu)
        if a <= 0:
            # Hyperbolic trajectory or numerically degenerate;
            # fall back to the body's IC.
            book_pos[:, body_i, :] = ric.pos[body_i]
            continue
        # Closed-form 2-body Kepler. M2 ≈ 0 keeps the primary
        # fixed in the perifocal frame; we translate by the
        # primary's IC to put it back in the global frame.
        kepler_traj = references.reference_kepler(
            M_primary, 1e-30, a, e, t, g=1.0)
        book_pos[:, body_i, :] = kepler_traj["pos"][:, 1, :] + primary_pos_0

    return book_pos


# ── Markdown report ─────────────────────────────────────────────────────────
def write_markdown_report(payloads: list[dict], out_path: str) -> None:
    """Write `real_case_report.md` summarising every preset's results."""
    lines = []
    lines.append("# Real-Case Validation Report\n")
    lines.append("Trained MLP / LSTM / GNN surrogates evaluated on real "
                 "Solar-System initial conditions. All numbers are in the "
                 "dimensionless N-body units the surrogates were trained on.\n")
    lines.append("## Out-of-distribution caveat\n")
    lines.append(
        "The surrogates were trained on 25-body synthetic galaxy discs "
        "(`simulation_3d.init_galaxy_disc`, mass ratio ≲ 10, body count = 25, "
        "Σm = 1, G = 1, no central sink). The Solar System is a *very* "
        "different distribution: 8-10 bodies with mass ratios of 10⁵ (Sun:Earth) "
        "or higher. The numbers below therefore measure **out-of-distribution "
        "generalisation**, not domain fit. The `disc_imf_in_distribution_baseline` "
        "preset provides an in-distribution sanity check for comparison.\n")
    lines.append("## Per-preset summary\n")

    # Reference key: the same legend shown at the top of every plot in
    # this report, rendered as a markdown table so a stranger reading
    # the .md alone (without the PNG) can map each surrogate to its
    # colour/linestyle in any of the per-preset `*_vs_book.png` /
    # `trajectory.png` / `energy.png`.
    lines.append("### Reading key\n")
    lines.append("Every line in the plots uses one of the styles below. "
                 "References are drawn in white. The book (closed-form "
                 "Kepler) line is green. Surrogates use a different "
                 "colour and linestyle per architecture:\n")
    lines.append("| line        | colour   | linestyle | meaning |")
    lines.append("|-------------|----------|-----------|---------|")
    lines.append("| book        | green    | solid     | Closed-form 2-body Kepler (primary + body, all other perturbations ignored) |")
    lines.append("| reference   | white    | solid     | Leapfrog at dt_ref = coarse dt / 100 |")
    lines.append("| GNN         | blue     | solid     | Trained GNN surrogate (`model_best.pt`) |")
    lines.append("| GNN_stable  | blue     | dashed    | GNN trained with stability loss (`model_best.pt` from `*/gnn_stable/`) |")
    lines.append("| LSTM        | orange   | dash-dot  | Trained LSTM surrogate (`model_best.pt` from `*/lstm/`) |")
    lines.append("| LSTM_stable | orange   | dotted    | LSTM trained with stability loss (`model_best.pt` from `*/lstm_stable/`) |")
    lines.append("| MLP         | violet   | dotted    | Trained MLP surrogate (`model_best.pt` from `*/mlp/`) |")
    lines.append("| MLP_stable  | violet   | densely dotted | MLP trained with stability loss (`model_best.pt` from `*/mlp_stable/`) |")
    lines.append("")

    for p in payloads:
        lines.append(f"### {p['name']}, {p['label']}\n")
        lines.append(f"- N = {p['n_bodies']}, samples = {p['n_samples']}, "
                     f"dt_N = {p['dt_N']:.3e}")
        if not p.get("in_distribution", False):
            sc = p["scale"]
            lines.append(f"- scale: M = {sc['M_kg']:.3e} kg, "
                         f"L = {sc['L_m']:.3e} m, T = {sc['T_s']:.3e} s")
        lines.append("")
        lines.append("| model | MSE (pos) | MSE (state) | max err / L | mean err / L | "
                     "frames to ½L error | max energy drift |")
        lines.append("|---|---|---|---|---|---|---|")
        for model_name, mm in p["per_model"].items():
            lines.append(
                f"| {model_name} | {mm['mse_position']:.3e} | "
                f"{mm['mse_state']:.3e} | "
                f"{mm['max_error_over_L']:.3e} | "
                f"{mm['mean_error_over_L']:.3e} | "
                f"{mm['frames_before_half_L']} | "
                f"{mm['max_energy_drift']:.3e} |"
            )
        lines.append("")

    # Append a Kepler-3rd-law section for every preset that has one
    # (in-distribution disc baseline is skipped, no clear primary).
    kepler_section_added = False
    for p in payloads:
        kepler_rows = p.get("kepler_check", [])
        if not kepler_rows:
            continue
        section = kepler_check.render_kepler_markdown(
            preset_name=p["name"], preset_label=p["label"],
            rows=kepler_rows, n_samples=p["n_samples"],
            dt_N=p["dt_N"], duration_years=p["duration_years"])
        if section:
            if not kepler_section_added:
                lines.append("## Kepler's 3rd-law check (reference integrator)\n")
                lines.append(
                    "For each preset we measure the orbital period T and "
                    "semi-major axis a of every non-primary body from the "
                    "reference trajectory, and compare T²/a³ to the "
                    "predicted 4π²/(G·M_primary). All bodies in the same "
                    "preset should give the same K = T²/a³ (that's the "
                    "law). The deviation is reported as a percentage. "
                    "Bodies that don't complete at least one full orbit "
                    "in the simulation window show NaN: increase "
                    "`duration_years` to bring them in.\n")
                kepler_section_added = True
            lines.append(section)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Driver ───────────────────────────────────────────────────────────────────
def _parse_ckpt(spec: str) -> tuple[str, str, str | None]:
    """Parse `path` | `path:type` | `path:type:name`.

    Returns (path, model_type, display_name|None). Default type is gnn.
    The optional third segment is a display name used to label this
    checkpoint in the report, so multiple same-type checkpoints can
    coexist without colliding on the model_type-derived key. Checkpoint
    paths are assumed not to contain ':' (true for the paths used here).
    """
    parts = spec.split(":")
    if len(parts) == 1:
        return parts[0], ModelType.GNN, None
    path = parts[0]
    kind = parts[1].strip().lower()
    if kind not in ModelType.values():
        raise ValueError(f"unknown model_type {kind!r}; expected one of {ModelType.values()}")
    name = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
    return path, kind, name


def main() -> None:
    p = argparse.ArgumentParser(
        description="Real-case validation of MLP/LSTM/GNN surrogates.")
    p.add_argument("--ckpt", action="append",
                   help="Checkpoint 'path.pt:type' (repeatable). "
                        "Default type is 'gnn' if missing. Optional "
                        "third segment sets the display name, so the same "
                        "type can be loaded twice with different labels. "
                        "Optional only when --list-presets is given.")
    p.add_argument("--preset", action="append",
                   help="Preset name or path to custom IC JSON (repeatable). "
                        "Default: all built-in presets.")
    p.add_argument("--preset-filter", choices=["inner", "outer", "galilean",
                                              "moon", "extended", "dist"],
                   help="Convenience filter — selects a named subset of "
                        "the built-in presets without listing every one. "
                        "`inner`=inner_planets, `outer`=full_solar_system, "
                        "`galilean`=jupiter_galileans, `moon`=sun_planets_moon, "
                        "`extended`=solar_system_extended, "
                        "`dist`=disc_imf_in_distribution_baseline.")
    p.add_argument("--list-presets", action="store_true",
                   help="Print every built-in preset name + label and exit.")
    p.add_argument("--out", default="real_case_validation/report",
                   help="Output directory. Each preset writes into a "
                        "preset_<name>/ subdir; the dashboard, markdown, "
                        "and JSON sit at the top level. Pass a different "
                        "--out per rerun to keep histories separate.")
    p.add_argument("--quick", action="store_true",
                   help="Smaller reference sub-stepping (faster, less precise).")
    p.add_argument("--single-step", action="store_true",
                   help="Run the single-step variant instead of the "
                        "autoregressive rollout. Each surrogate predicts "
                        "the next frame only (no compounding); the "
                        "report is the 1-3 %% single-step MSE rather "
                        "than the rollout-averaged error. Writes to "
                        "<out>/single_step/ and a single-step markdown "
                        "summary.")
    p.add_argument("--warmup", type=int, default=None,
                   help=(f"Override the inference warm-up window. The "
                         f"surrogates were trained on W={REAL_CASE_WARMUP_WINDOW}; "
                         f"at inference you can build a longer warm-up from "
                         f"the leapfrog reference to lower single-step error. "
                         f"Must be between 5 and 50 (capped at n_samples // 4). "
                         f"Default: {REAL_CASE_WARMUP_WINDOW} (no change)."))
    p.add_argument("--ensemble", default=None,
                   help="Comma-separated model names to ensemble at "
                        "inference (e.g. 'lstm,gnn'). The ensemble's "
                        "prediction is the per-component mean of the "
                        "listed models' predictions. The ensemble is "
                        "written into summary.json as a synthetic "
                        "ensemble_<list> row. Default: no ensemble.")
    p.add_argument("--dump-preds", action="store_true",
                   help="Persist the per-model rollout arrays to "
                        "preset_dir/preds.npy + book_pos.npy + "
                        "preds_meta.json so external tools (e.g. the "
                        "animation script) can rebuild visualisations "
                        "without re-running the surrogates.")
    args = p.parse_args()

    # Validate --warmup up-front (range is loose; each preset enforces
    # its own n_samples // 4 cap at runtime).
    if args.warmup is not None:
        if args.warmup < 5:
            sys.exit(f"[runner] --warmup must be >= 5 (got {args.warmup})")
        if args.warmup > 50:
            sys.exit(f"[runner] --warmup must be <= 50 (got {args.warmup})")

    # Validate --ensemble up-front: parse and store the canonical
    # display-name form ("LSTM" rather than "lstm") for downstream use.
    ensemble_models: list[str] | None = None
    if args.ensemble:
        type_to_display = {"mlp": "MLP", "mlp_stable": "MLP_stable",
                           "lstm": "LSTM", "lstm_stable": "LSTM_stable",
                           "gnn": "GNN", "gnn_stable": "GNN_stable"}
        ensemble_models = []
        for raw in args.ensemble.split(","):
            raw = raw.strip()
            if not raw:
                continue
            key = raw.lower()
            if key not in type_to_display:
                sys.exit(f"[runner] --ensemble: unknown model {raw!r}; "
                         f"must be one of {list(type_to_display)}")
            ensemble_models.append(type_to_display[key])
        if len(ensemble_models) < 2:
            sys.exit("[runner] --ensemble needs at least 2 model names")

    # The list-presets flag should work even with no --ckpt (and even on
    # a machine that has no torch installed).
    if args.list_presets:
        _print_preset_catalog()
        return

    if not args.ckpt:
        sys.exit("[runner] at least one --ckpt is required (or pass "
                 "--list-presets to print the catalog).")

    device = pick_device()
    print(f"[device] {device}")

    # Load every checkpoint.
    loaded: list[LoadedModel] = []
    for spec in args.ckpt:
        path, kind, name = _parse_ckpt(spec)
        # Resolve paths against the caller's cwd so the user can run the
        # command from anywhere and not have to cd into the repo root.
        path = str(Path(path).expanduser().resolve())
        print(f"[ckpt] {kind:5s} <- {path}" + (f"  (label: {name})" if name else ""))
        try:
            loaded.append(load_model(path, kind, device, display_name=name))
        except Exception as e:
            print(f"  ! load failed: {e!r}")
    if not loaded:
        sys.exit("[runner] no checkpoints loaded; aborting.")

    # Decide which presets to run.
    if args.preset:
        preset_specs = list(args.preset)
    elif args.preset_filter:
        preset_specs = _PRESET_FILTER_MAP[args.preset_filter]
        print(f"[filter] {args.preset_filter}: "
              f"{preset_specs}")
    else:
        from . import presets as _p
        preset_specs = [pp["name"] for pp in _p.PRESETS]

    out_path = Path(args.out).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Single-step mode: predict the next frame only (no autoregressive
    # compounding). Writes to <out>/single_step/.
    if args.single_step:
        if args.dump_preds:
            print("[runner] --dump-preds: single-step arrays will be "
                  "written to <out>/single_step/preset_*/ "
                  "(preds.npy, ref_pos.npy, preds_meta.json).")
        ss_out = out_path / "single_step"
        ss_out.mkdir(parents=True, exist_ok=True)
        ss_payloads: list[dict] = []
        for spec in preset_specs:
            try:
                ss_payloads.append(run_single_step(
                    spec, loaded, device, str(ss_out),
                    quick=args.quick,
                    warmup=args.warmup,
                    dump_preds=args.dump_preds))
            except Exception as e:
                print(f"[runner] single-step preset {spec!r} failed: {e!r}")
        ss_md = ss_out / "single_step_report.md"
        write_single_step_markdown(ss_payloads, str(ss_md))
        print(f"[single-step] {ss_md}")
        # JSON dump with the per-model arrays removed.
        ss_json = ss_out / "single_step_report.json"
        with open(ss_json, "w", encoding="utf-8") as f:
            json.dump(ss_payloads, f, indent=2)
        print(f"[single-step] {ss_json}")
        return

    payloads: list[dict] = []
    for spec in preset_specs:
        try:
            payloads.append(run_preset(spec, loaded, device, str(out_path),
                                       quick=args.quick,
                                       warmup=args.warmup,
                                       ensemble=ensemble_models,
                                       dump_preds=args.dump_preds))
        except Exception as e:
            print(f"[runner] preset {spec!r} failed: {e!r}")

    # Dashboard
    dash_payloads = []
    for p in payloads:
        dash_payloads.append({
            "name":                p["name"],
            "ref_pos":             p["_ref_pos"],
            "ref_energy":          p["_ref_energy"],
            "surrogate_trajs":     {k: v[..., :3]
                                    for k, v in p["_surrogate_trajs"].items()},
            "surrogate_energies":  p["_surrogate_energies"],
        })
    if dash_payloads:
        dash_path = out_path / "dashboard.png"
        rv_plots.plot_dashboard(dash_payloads, str(dash_path))
        print(f"[dash] {dash_path}")

    # Markdown report
    md_path = out_path / "real_case_report.md"
    write_markdown_report(payloads, str(md_path))
    print(f"[md]   {md_path}")

    # JSON report (drop the in-memory arrays).
    json_payloads = []
    for p in payloads:
        json_payloads.append({k: v for k, v in p.items()
                              if not k.startswith("_")})
    json_path = out_path / "real_case_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payloads, f, indent=2)
    print(f"[json] {json_path}")


# ── Preset catalog ───────────────────────────────────────────────────────────
# Convenience subsets used by the `--preset-filter` flag. Keeping the
# exact slugs here means a stray rename in `presets.PRESETS` will fail
# loudly at import time (`KeyError`) rather than silently dropping a
# preset from the user's filter.
_PRESET_FILTER_MAP: dict[str, list[str]] = {
    "inner":    ["inner_planets"],
    "outer":    ["full_solar_system"],
    "galilean": ["jupiter_galileans"],
    "moon":     ["sun_planets_moon"],
    "extended": ["solar_system_extended"],
    "dist":     ["disc_imf_in_distribution_baseline"],
}


def _print_preset_catalog() -> None:
    """Print every built-in preset name + label and exit."""
    from . import presets as _p
    print("Built-in real-case presets:")
    print("=" * 72)
    for p in _p.PRESETS:
        n = p.get("name", "?")
        l = p.get("label", "")
        print(f"  {n:<40s}  {l}")
    print("=" * 72)
    print("Convenience filters (--preset-filter):")
    for k, v in _PRESET_FILTER_MAP.items():
        print(f"  {k:<10s}  -> {', '.join(v)}")


if __name__ == "__main__":
    main()