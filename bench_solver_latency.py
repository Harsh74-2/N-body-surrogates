#!/usr/bin/env python3
"""
bench_solver_latency.py
=======================
Closes the latency-crossover gap in the latency benchmark, honestly.

It measures, on the *local* CPU (no VM needed), the per-frame latency of:

  1. The direct O(N^2) N-body solver  -- one leapfrog (Stoermer-Verlet) step
     advances the whole N-body system by one frame.
  2. The three neural surrogates (MLP, LSTM, GNN) -- one forward pass predicts
     the next state for N bodies. The N=10 *fixed* per-body checkpoints are
     used; because the architectures are per-body / shared-weight, a single
     checkpoint runs at any N, which is exactly the variable-N transfer we
     want to demonstrate for latency.

Two regimes are reported, because the answer to "when is the surrogate
cheaper?" depends on how it is run:

  * single-frame (B=1)   -- one frame at a time. Worst case for the surrogate,
    because each call pays the full PyTorch dispatch overhead. This is the
    apples-to-apples comparison to the solver, which advances one system
    per call.
  * batched (B=64 amortised) -- per-frame cost = (batch wall time)/B. This is
    the throughput regime the surrogate is actually built for, and matches
    how evaluate_models.py reports latency (ms/batch). The per-call overhead
    amortises across the batch.

Outputs:
    results/latency_bench.json
    plots/scaling_latency.png        (2 panels: single-frame vs batched; also
                                      copied to report_figures/)

Usage:
    python bench_solver_latency.py
    python bench_solver_latency.py --n 10 25 50 100 200 --repeats 40

Checkpoints are expected under the sweep layout training_runs/N{n}/{mlp,lstm,gnn}/model_best.pt.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from utils import configure_utf8_stdout

configure_utf8_stdout()

import numpy as np
import torch

from pipeline_config import (
    DEFAULT_EPS,
    DEFAULT_GRAVITY_G,
    EVAL_BATCH_SIZE,
    RESULTS_DIR,
    SWEEP_N_VALUES,
)
from simulation_3d import compute_accelerations, leapfrog_step

MODEL_COLOR = {"mlp": "#1f77b4", "lstm": "#ff7f0e", "gnn": "#2ca02c"}
MODEL_MARK  = {"mlp": "o",       "lstm": "s",        "gnn": "^"}
SOLVER_COLOR = "#000000"

CKPT_PATHS = {
    "mlp":  Path("training_runs/N10/mlp/model_best.pt"),
    "lstm": Path("training_runs/N10/lstm/model_best.pt"),
    "gnn":  Path("training_runs/N10/gnn/model_best.pt"),
}
WINDOW = 5
FEAT = 6
DT = 0.01
BATCH = 64          # batched-regime batch size (matches EVAL_BATCH_SIZE)
BATCH_FALLBACK = [64, 32, 16, 8, 4]   # halved on memory error


# ── Solver timing ────────────────────────────────────────────────────────────
def time_solver(n: int, repeats: int, warmup: int = 3) -> float:
    """Mean ms per leapfrog step (one frame) for an N-body system."""
    rng = np.random.default_rng(n)
    pos = rng.standard_normal((n, 3))
    vel = rng.standard_normal((n, 3)) * 0.1
    mass = np.full(n, 1.0 / n)
    acc = compute_accelerations(pos, mass, DEFAULT_EPS, g=DEFAULT_GRAVITY_G)

    for _ in range(warmup):
        pos, vel, acc = leapfrog_step(pos, vel, acc, mass, DT, DEFAULT_EPS, g=DEFAULT_GRAVITY_G)

    ts: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(5):
            pos, vel, acc = leapfrog_step(pos, vel, acc, mass, DT, DEFAULT_EPS, g=DEFAULT_GRAVITY_G)
        ts.append((time.perf_counter() - t0) * 1000.0 / 5.0)
    return float(statistics.fmean(ts))


# ── Surrogate timing ────────────────────────────────────────────────────────
def _load_model(model_type: str, device: torch.device):
    from evaluate_models import build_model
    p = CKPT_PATHS[model_type]
    if not p.is_file():
        return None
    return build_model(str(p), model_type, device)


def _forward(model, model_type: str, x, mass):
    if model_type == "gnn":
        return model(x, mass=mass)
    return model(x, mass)


def time_surrogate_batch(model, model_type: str, n: int, batch: int,
                         repeats: int, warmup: int = 3,
                         device: torch.device = torch.device("cpu")) -> float:
    """Mean ms per frame, amortised over a batch of `batch` frames."""
    g = torch.Generator().manual_seed(n)
    x = torch.randn(batch, WINDOW, n, FEAT, generator=g)
    mass = torch.rand(batch, n, generator=g) + 0.1
    x, mass = x.to(device), mass.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _forward(model, model_type, x, mass)
        ts: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _forward(model, model_type, x, mass)
            ts.append((time.perf_counter() - t0) * 1000.0 / batch)
    return float(statistics.fmean(ts))


def _batch_for(n: int, model_type: str) -> list[int]:
    """Pick a batch-size fallback list that fits memory and time at this N."""
    if n <= 50:
        return [64, 32, 16]
    if n <= 100:
        return [32, 16, 8] if model_type == "gnn" else [64, 32, 16]
    # n == 200: the GNN message tensor (B, N, N, 2h+1) is large; keep B small.
    return [8, 4, 2] if model_type == "gnn" else [32, 16, 8]


def time_surrogate(model, model_type: str, n: int, repeats: int,
                   device: torch.device = torch.device("cpu")) -> tuple[float | None, float | None]:
    """Return (single_frame_ms, batched_amortised_ms). None if it failed."""
    try:
        single = time_surrogate_batch(model, model_type, n, 1, max(repeats, 8), device=device)
    except Exception:
        single = None
    batched = None
    for B in _batch_for(n, model_type):
        try:
            batched = time_surrogate_batch(model, model_type, n, B, max(repeats, 6), device=device)
            break
        except RuntimeError as e:        # OOM -> smaller batch
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                continue
            batched = None
            break
        except Exception:
            batched = None
            break
    return single, batched


def _saved_surrogate_latency(model_type: str, n: int) -> float | None:
    """Fallback: ms/batch from results/N{n}/metrics.json -> ms/frame (amortised)."""
    p = Path(RESULTS_DIR) / f"N{n}" / "metrics.json"
    if not p.is_file():
        return None
    recs = json.loads(p.read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else [recs]
    for r in recs:
        if r.get("model_type", "").lower() == model_type and r.get("latency_ms") is not None:
            ns, nb = r.get("n_samples"), r.get("n_batches")
            batch = (ns / nb) if (ns and nb) else EVAL_BATCH_SIZE
            return float(r["latency_ms"]) / max(batch, 1)
    return None


# ── Crossover ────────────────────────────────────────────────────────────────
def crossover(ns, solver, surr) -> int | None:
    for n in ns:
        if solver.get(n) is not None and surr.get(n) is not None and surr[n] < solver[n]:
            return n
    return None


# ── Plot ─────────────────────────────────────────────────────────────────────
def _panel(ax, ns, solver, surrogates, crossovers, title: str, ylab: str) -> None:
    xs = [n for n in ns if solver.get(n) is not None]
    ax.plot(xs, [solver[n] for n in xs], "--", color=SOLVER_COLOR, marker="D",
            markersize=6, linewidth=2, label="Direct solver  O(N$^2$)")
    for mt in ("mlp", "lstm", "gnn"):
        xs = [n for n in ns if surrogates[mt].get(n) is not None]
        if not xs:
            continue
        ax.plot(xs, [surrogates[mt][n] for n in xs], "-", color=MODEL_COLOR[mt],
                marker=MODEL_MARK[mt], markersize=7, linewidth=2,
                markerfacecolor="white", markeredgewidth=1.8, label=f"{mt.upper()} surrogate")
        nstar = crossovers.get(mt)
        if nstar is not None and surrogates[mt].get(nstar) is not None:
            ax.axvline(nstar, color=MODEL_COLOR[mt], ls=":", lw=1.0, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("body count N", fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)


def make_plot(ns, solver, single, batched, cross_single, cross_batched, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    _panel(axes[0], ns, solver, single, cross_single,
           "(a) Single-frame latency (B=1, CPU)", "ms / frame")
    _panel(axes[1], ns, solver, batched, cross_batched,
           f"(b) Batched throughput, amortised (B$\\leq${BATCH}, CPU)", "ms / frame (amort.)")
    fig.suptitle("Per-frame latency: direct solver vs. neural surrogates", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Latency crossover: direct solver vs surrogates.")
    ap.add_argument("--n", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--out", default="plots/scaling_latency.png")
    ap.add_argument("--json", default=f"{RESULTS_DIR}/latency_bench.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    ns = args.n

    models: dict[str, torch.nn.Module | None] = {}
    for mt in ("mlp", "lstm", "gnn"):
        models[mt] = _load_model(mt, device)
        print(f"[load] {mt}: {'OK' if models[mt] is not None else 'checkpoint missing -- will use saved latency'}")

    solver: dict[int, float] = {}
    single: dict[str, dict[int, float | None]] = {"mlp": {}, "lstm": {}, "gnn": {}}
    batched: dict[str, dict[int, float | None]] = {"mlp": {}, "lstm": {}, "gnn": {}}

    print(f"\n{'N':>5} {'solver':>9} | {'MLP sgl':>8} {'LSTM sgl':>8} {'GNN sgl':>8} | "
          f"{'MLP b64':>8} {'LSTM b64':>8} {'GNN b64':>8}  (ms/frame)")
    print("-" * 86)
    for n in ns:
        s = time_solver(n, args.repeats)
        solver[n] = s
        row = [f"{n:>5}", f"{s:>9.4f}", "|"]
        for mt in ("mlp", "lstm", "gnn"):
            if models[mt] is not None:
                sg, bt = time_surrogate(models[mt], mt, n, args.repeats, device=device)
            else:
                sg = None
                bt = _saved_surrogate_latency(mt, n) if n in SWEEP_N_VALUES else None
            single[mt][n] = sg
            batched[mt][n] = bt
            row.append(f"{sg:>8.4f}" if sg is not None else f"{'--':>8}")
        row.append("|")
        for mt in ("mlp", "lstm", "gnn"):
            bt = batched[mt][n]
            row.append(f"{bt:>8.4f}" if bt is not None else f"{'--':>8}")
        print(" ".join(row))

    cs = {mt: crossover(ns, solver, single[mt]) for mt in ("mlp", "lstm", "gnn")}
    cb = {mt: crossover(ns, solver, batched[mt]) for mt in ("mlp", "lstm", "gnn")}
    print("\nCrossover N* (smallest N where surrogate < solver):")
    print(f"  {'model':>6}  {'single-frame':>13}  {'batched':>8}")
    for mt in ("mlp", "lstm", "gnn"):
        print(f"  {mt.upper():>6}  {str(cs[mt]):>13}  {str(cb[mt]):>8}")

    out = Path(args.out)
    make_plot(ns, solver, single, batched, cs, cb, out)
    if Path("report_figures").is_dir():
        import shutil
        shutil.copy(out, Path("report_figures") / out.name)
        print(f"[plot] copied report_figures/{out.name}")

    payload = {
        "n_values": ns, "repeats": args.repeats, "device": "cpu",
        "batch_size": BATCH,
        "solver": {str(k): v for k, v in solver.items()},
        "surrogate_single_frame": {mt: {str(k): v for k, v in d.items()} for mt, d in single.items()},
        "surrogate_batched_amortised": {mt: {str(k): v for k, v in d.items()} for mt, d in batched.items()},
        "crossover_Nstar_single_frame": cs,
        "crossover_Nstar_batched": cb,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[json] saved {args.json}")


if __name__ == "__main__":
    main()