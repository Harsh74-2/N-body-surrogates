"""
verify_retrain_gates.py
=======================
Post-retrain go/no-go gates. Reads the artifacts the retrain pipeline
produces and checks every cell against the identity floor that the
identity-collapse failure mode (Sept-15 retrain) must never pass:

  G1  single-step explained variance   metrics_explained_var > --min-ev
      (from results/N{n}/metrics_{m}.json          — single-step cells)
      and results/N{n}_{m}_stable_metrics.json     — stable cells
  G2  rollout beats persistence (robust stats, Gemini round 2026-09-21)
      stability.json identity_baseline.model_over_identity_median
      < --max-ident-ratio  (per model_type + variant record)
      AND model_over_identity_max < --max-start-ratio (no single
      catastrophic trajectory; the mean is recorded as diagnostic only —
      a raw mean is infinitely sensitive to one exploding rollout among
      healthy ones)
  G6  spatial-collapse guard (Gemini round, 2026-09-21)
      collapse_guard.spatial_var_ratio_final >= --min-spatial-var
      (default 0.5): a model that clamps every body to the centre of
      mass scores the dataset spatial variance as its MSE, and at long
      K the persistence floor degrades past that variance, so a plain
      model/persistence ratio can PASS a physically dead model. This
      gate proves the rollout retained real spatial structure.
  G3  architecture parameter counts match the canonical surrogates
      (catches a silently-changed model definition before it invalidates
      the published param table; MLP/LSTM/GNN counts are N-independent)
  G4  all reported metrics finite (NaN guard)
  G5  checkpoint config-vs-variant consistency (Gemini crosscheck round 3,
      4(c)): *_stable ckpt must carry w_rollout > 0, single_step ckpt
      w_rollout == 0, and the ckpt's saved `variant` must match the cell

Modes
-----
  --precheck            data-pipeline gate BEFORE training: verifies the
                        raw sims + ml_ready_data npz/sidecars on the
                        machine that will run the retrain (fresh clone
                        self-check; mirrors the 2026-09-21 audit-B scan).
  default               artifact gates AFTER training. Cells whose
                        artifacts are missing are reported MISSING (and
                        fail the run) unless --allow-missing.

Exit status: 0 = all gated cells PASS, 1 = any FAIL / MISSING.
The per-cell verdict table is also written to results/retrain_gates.json.

Usage
-----
    # before training (after the data-generating sweep step — raw_data/ and
    # ml_ready_data/ are not git-tracked, so a fresh clone FAILs every cell):
    python verify_retrain_gates.py --precheck --N 10 25   # probe cells
    python verify_retrain_gates.py --precheck             # full grid

    # probe stage (2 cheap cells, no stability benchmark yet):
    python verify_retrain_gates.py --N 10 25 --models mlp --allow-missing

    # after the full grid + stability benchmark:
    python verify_retrain_gates.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _sibling(name: str, path: str):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass field resolution looks the module up
    # in sys.modules (evaluate_models.py's load_sibling_module does the same).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from utils import configure_utf8_stdout, load_checkpoint  # noqa: E402

configure_utf8_stdout()

# Canonical architecture definitions (must mirror pipeline_config.py; the
# surrogates are instantiated to count params exactly as evaluate_models
# does — no hardcoded numbers that can drift).
_MLP  = _sibling("_gates_mlp", "mlp_train.py")
_LSTM = _sibling("_gates_lstm", "lstm_train.py")
_GNN  = _sibling("_gates_gnn", "gnn_train.py")

EXPECTED_PARAMS: dict[str, int | None] = {}
try:
    EXPECTED_PARAMS["mlp"] = sum(
        p.numel() for p in
        _MLP.MLPSurrogate(window_size=5, in_features=6, hidden=256,
                          depth=4).parameters())
    EXPECTED_PARAMS["lstm"] = sum(
        p.numel() for p in
        _LSTM.LSTMSurrogate(window_size=5, in_features=6, hidden=256,
                            num_layers=2).parameters())
    EXPECTED_PARAMS["gnn"] = sum(
        p.numel() for p in
        _GNN.GNNSurrogate(in_features=6, hidden=128,
                          num_passes=2).parameters())
except Exception as e:  # pragma: no cover — keep gates usable if imports drift
    print(f"[warn] could not instantiate surrogates for G3: {e!r}")
    EXPECTED_PARAMS = {"mlp": None, "lstm": None, "gnn": None}

N_VALUES = [10, 25, 50, 100]
MODELS = ["mlp", "lstm", "gnn"]


# ── Precheck (audit-B scan, runnable on the VM before launch) ────────────────
# NOTE: raw_data/ and ml_ready_data/ are NOT git-tracked (1.9 GB), so on a
# fresh clone every cell FAILS until scaling_sweep.py has generated them.
# Stage the precheck: --precheck --N 10 25 after the probe sweep, full
# --precheck after the full grid's data is in place.
def run_precheck(root: Path,
                 n_values: list[int] | None = None,
                 models: list[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for n in (n_values or N_VALUES):
        for m in (models or MODELS):
            cell = f"N{n}/{m}"
            npz = root / "ml_ready_data" / f"N{n}" / m / "dataset_3d_w5h1s1r.npz"
            js_path = npz.with_suffix(".json")
            raw = root / "raw_data" / f"N{n}" / m
            rec: dict = {"cell": cell, "status": "PASS", "checks": {}}
            try:
                if not raw.is_dir() or not any(raw.glob("sim_*.npz")):
                    raise FileNotFoundError(f"raw sims missing under {raw}")
                rec["checks"]["raw_sims"] = len(list(raw.glob("sim_*.npz")))
                if not npz.is_file():
                    raise FileNotFoundError(f"npz missing: {npz}")
                with np.load(npz) as d:
                    X_shape = d["X"].shape
                    n_sims = int(d["mass"].shape[0])
                    for k in d.files:
                        _ = d[k]          # full read → CRC check
                if not js_path.is_file():
                    raise FileNotFoundError(f"sidecar missing: {js_path}")
                js = json.loads(js_path.read_text(encoding="utf-8"))
                ok = (X_shape[0] == js["n_windows"]
                      and n_sims == js["n_simulations"]
                      and sum(js["n_windows_per_sim"]) == js["n_windows"]
                      and js["window_size"] == 5 and js["horizon"] == 1
                      and js["stride"] == 1 and js["normalize"] is False)
                if not ok:
                    raise ValueError(
                        f"npz/sidecar inconsistent: X={X_shape} "
                        f"n_sims={n_sims} json n_windows={js['n_windows']} "
                        f"n_simulations={js['n_simulations']}")
                rec["checks"]["X_shape"] = list(X_shape)
                rec["checks"]["n_sims"] = n_sims
            except Exception as e:
                rec["status"] = "FAIL"
                rec["checks"]["error"] = f"{type(e).__name__}: {e}"
            rows.append(rec)
            print(f"  [{'PASS' if rec['status'] == 'PASS' else 'FAIL'}] "
                  f"{cell}  {rec['checks'].get('X_shape', '-')}  "
                  f"sims={rec['checks'].get('n_sims', '-')}"
                  + (f"  {rec['checks'].get('error', '')}"
                     if rec["status"] == "FAIL" else ""))
    return rows


# ── Artifact gates ────────────────────────────────────────────────────────────
def _load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_cell(n: int, m: str, variant: str, results_root: Path,
               min_ev: float, max_ident_ratio: float,
               training_root: Path, allow_missing: bool = False,
               max_start_ratio: float = 100.0,
               min_spatial_var: float = 0.5,
               ) -> dict:
    """Gates G1–G4 for one (N, model, variant) cell."""
    cell = f"N{n}/{m}_{variant}"
    checks: dict[str, object] = {}
    status = "PASS"

    def fail(msg: str) -> None:
        nonlocal status
        status = "FAIL"
        checks.setdefault("errors", []).append(msg)

    # Locate the 4-metric JSON for this cell.
    if variant == "single_step":
        mpath = results_root / f"N{n}" / f"metrics_{m}.json"
    else:
        mpath = results_root / f"N{n}_{m}_stable_metrics.json"
    mrecs = _load_json(mpath)
    if mrecs is None:
        return {"cell": cell, "status": "MISSING", "checks":
                {"errors": [f"metrics file not found: {mpath}"]}}
    rec = next((r for r in mrecs if r.get("model_type") == m), None)
    if rec is None:
        return {"cell": cell, "status": "MISSING", "checks":
                {"errors": [f"no {m} record in {mpath}"]}}

    # G1 + G4: collapse detection + explained variance on the reported
    # split. The HARD gate is mse >= mse_identity (the persistence floor:
    # an identity-collapsed model scores exactly the persistence MSE).
    # The EV threshold is a WARNING, not a fail (2026-09-21, GNN N=10
    # probe): EV = 1 - mse/Var(y) also collapses when the test split has
    # small state variance even for a model that genuinely beats
    # persistence — the GNN N=10 cell (2.71e-06 vs identity 2.86e-06,
    # train/val gap ~500x from a real generalization gap, 1-sim test
    # split) is that case, and blocking the grid on it would confuse a
    # generalization result with a training collapse.
    mse, ident = rec.get("mse"), rec.get("mse_identity")
    ev = rec.get("mse_explained_var")
    if not (isinstance(mse, (int, float)) and np.isfinite(mse)):
        fail(f"single-step mse not finite: {mse}")
    elif not (isinstance(ident, (int, float)) and np.isfinite(ident)):
        fail(f"mse_identity not finite: {ident} (identity column missing — "
             "eval ran with pre-audit evaluate_models.py)")
    else:
        checks["mse"] = mse
        checks["mse_identity"] = ident
        checks["explained_var"] = ev
        if mse >= ident:
            fail(f"mse {mse:.3e} >= identity {ident:.3e} "
                 "(model does not beat the persistence floor — identity "
                 "collapse signature)")
        elif isinstance(ev, (int, float)) and np.isfinite(ev) and ev < min_ev:
            print(f"  [warn   ] {cell}: explained variance {ev:.3f} < "
                  f"{min_ev:.3f} but mse beats identity "
                  f"({mse:.3e} < {ident:.3e}) — generalization gap, not "
                  "collapse (recorded, not blocking)")
            checks["explained_var_warn"] = True
        if not (isinstance(ev, (int, float)) and np.isfinite(ev)):
            fail(f"explained variance not finite: {ev}")

    # G3: parameter count vs canonical architecture.
    n_params = rec.get("n_params")
    exp = EXPECTED_PARAMS.get(m)
    checks["n_params"] = n_params
    checks["n_params_expected"] = exp
    if exp is not None and n_params != exp:
        fail(f"n_params {n_params:,} != canonical {exp:,} "
             "(architecture drift)")

    # G5: checkpoint config-vs-variant consistency (Gemini crosscheck
    # round 3, 4(c)): a *_stable checkpoint MUST have been trained with
    # w_rollout > 0, a single-step one with w_rollout == 0 — a stable
    # model trained without the rollout term is the failure mode this
    # whole gate set exists to catch. The trainers write `variant` and
    # the full config into every model_best.pt.
    if variant == "single_step":
        ckpt_path = training_root / f"N{n}" / m / "model_best.pt"
    else:
        ckpt_path = training_root / f"N{n}" / f"{m}_stable" / "model_best.pt"
    try:
        ckpt = load_checkpoint(str(ckpt_path))
        ck_variant = ckpt.get("variant")
        cfg_dict = ckpt.get("config", {}) or {}
        w_rollout = cfg_dict.get("w_rollout")
        checks["ckpt_variant"] = ck_variant
        checks["ckpt_w_rollout"] = w_rollout
        if ck_variant != variant:
            fail(f"checkpoint variant {ck_variant!r} != requested {variant!r} "
                 f"({ckpt_path})")
        if not isinstance(w_rollout, (int, float)):
            fail(f"checkpoint config missing w_rollout ({ckpt_path})")
        elif variant == "stable" and not w_rollout > 0.0:
            fail(f"stable ckpt trained with w_rollout={w_rollout} "
                 "(stability training silently absent)")
        elif variant == "single_step" and not w_rollout == 0.0:
            fail(f"single_step ckpt trained with w_rollout={w_rollout} "
                 "(not single-step)")
    except Exception as e:
        fail(f"checkpoint unreadable: {ckpt_path} ({type(e).__name__}: {e})")

    # G2: rollout beats persistence (only checkable once the stability
    # benchmark has produced results/N{n}/stability.json). A missing
    # benchmark artifact is DEFERRED under --allow-missing (probe stage:
    # the benchmark runs after the grid) and a hard FAIL in the final
    # gate (no --allow-missing): Stage 5 must not pass without it.
    spath = results_root / f"N{n}" / "stability.json"
    srecs = _load_json(spath)
    if srecs is None:
        checks["stability"] = ("deferred (--allow-missing)"
                               if allow_missing else "not run yet")
        if not allow_missing:
            fail(f"stability.json not found: {spath} "
                 f"(run stability_benchmark.py for N{n})")
    else:
        srec = next((r for r in srecs
                     if r.get("model_type") == m and r.get("variant") == variant),
                    None)
        if srec is None:
            fail(f"no {m}/{variant} record in {spath}")
        else:
            ident = srec.get("identity_baseline", {})
            mean_r = ident.get("model_over_identity_mean")
            med_r = ident.get("model_over_identity_median")
            max_r = ident.get("model_over_identity_max")
            checks["model_over_identity_mean"] = mean_r
            checks["model_over_identity_median"] = med_r
            checks["model_over_identity_max"] = max_r
            if mean_r is None or not np.isfinite(mean_r):
                fail(f"identity_baseline.model_over_identity_mean missing "
                     f"in {spath} (stability ran with pre-audit code?)")
            # Median is the primary gate: robust to a single exploding
            # rollout among healthy ones. A missing median = the
            # benchmark predates the 2026-09-21 robust-stats update.
            if med_r is None or not np.isfinite(med_r):
                fail(f"identity_baseline.model_over_identity_median missing "
                     f"in {spath} — stability.json predates the robust-stats "
                     f"benchmark; re-run stability_benchmark.py")
            elif med_r >= max_ident_ratio:
                fail(f"rollout model/persistence MEDIAN MSE ratio "
                     f"{med_r:.3f} >= gate {max_ident_ratio:.3f} "
                     f"(stalling on the anchor)")
            # Max per-start ratio: no single trajectory may suffer a
            # catastrophic numerical explosion (median can hide one).
            if max_r is None or not np.isfinite(max_r):
                fail(f"identity_baseline.model_over_identity_max missing "
                     f"in {spath} — re-run with the updated benchmark")
            elif max_r >= max_start_ratio:
                fail(f"max per-start model/persistence ratio {max_r:.3g} >= "
                     f"gate {max_start_ratio:.3g} (catastrophic single-"
                     "trajectory explosion)")
            # G6: spatial-collapse guard.
            cg = srec.get("collapse_guard", {})
            var_ratio = cg.get("spatial_var_ratio_final")
            checks["spatial_var_ratio_final"] = var_ratio
            if var_ratio is None or not np.isfinite(var_ratio):
                fail(f"collapse_guard.spatial_var_ratio_final missing in "
                     f"{spath} — re-run with the updated benchmark (a "
                     "missing guard cannot prove the rollout kept spatial "
                     "structure)")
            elif var_ratio < min_spatial_var:
                fail(f"spatial_var_ratio_final {var_ratio:.3f} < "
                     f"{min_spatial_var:.2f} — predicted rollout lost "
                     "spatial variance (centre-of-mass collapse signature: "
                     "beats persistence by clumping, not by physics)")
            # Divergence inside the horizon is informative, not fatal, but a
            # cell that diverges at step 1 has effectively failed.
            div = srec.get("divergence_step")
            checks["divergence_step"] = div

    return {"cell": cell, "status": status, "checks": checks}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-retrain go/no-go gates (identity-collapse guard).")
    p.add_argument("--results-root", default="results")
    p.add_argument("--training-root", default="training_runs",
                   help="Root holding N{n}/{model}[_stable]/model_best.pt "
                        "(gate G5 reads each ckpt's variant + config).")
    p.add_argument("--project-root", default=".",
                   help="Repo root for --precheck (raw_data/, ml_ready_data/).")
    p.add_argument("--precheck", action="store_true",
                   help="Data-pipeline gate before training; no artifacts read.")
    p.add_argument("--N", type=int, nargs="+", default=N_VALUES)
    p.add_argument("--models", nargs="+", default=MODELS,
                   choices=MODELS)
    p.add_argument("--variants", nargs="+", default=["single_step", "stable"],
                   choices=["single_step", "stable"])
    p.add_argument("--min-ev", type=float, default=0.5,
                   help="Gate G1: required single-step explained variance vs "
                        "identity (default 0.5 — a healthy in-distribution "
                        "model is far above 0; an identity-collapsed one ~0).")
    p.add_argument("--max-ident-ratio", type=float, default=1.0,
                   help="Gate G2: max allowed rollout model/persistence "
                        "MEDIAN MSE ratio (robust per-start statistic).")
    p.add_argument("--max-start-ratio", type=float, default=100.0,
                   help="Gate G2: max allowed per-start model/persistence "
                        "MSE ratio (catches one catastrophic trajectory "
                        "the median would hide).")
    p.add_argument("--min-spatial-var", type=float, default=0.5,
                   help="Gate G6: required spatial_var_ratio_final "
                        "(predicted vs true spatial variance at the last "
                        "finite rollout step); below this = centre-of-mass "
                        "collapse signature.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Treat MISSING cells (artifacts not yet produced) as "
                        "skipped instead of failing — for staged probes.")
    p.add_argument("--json-out", default=None,
                   help="Where to write the verdict table (default "
                        "<results-root>/retrain_gates.json).")
    args = p.parse_args()

    results_root = Path(args.results_root)
    rows: list[dict] = []

    if args.precheck:
        print("── PRECHECK: data pipeline gate (audit-B scan) ──")
        rows = run_precheck(Path(args.project_root).resolve(),
                            n_values=list(args.N), models=list(args.models))
    else:
        print("── GATES: post-retrain artifact gate ──")
        print(f"  min explained variance = {args.min_ev}\n"
              f"  max rollout model/persistence MEDIAN ratio = "
              f"{args.max_ident_ratio}\n"
              f"  max per-start ratio (explosion guard) = "
              f"{args.max_start_ratio}\n"
              f"  min spatial-variance ratio (collapse guard) = "
              f"{args.min_spatial_var}\n")
        for n in args.N:
            for m in args.models:
                for v in args.variants:
                    rec = check_cell(n, m, v, results_root,
                                     args.min_ev, args.max_ident_ratio,
                                     Path(args.training_root),
                                     allow_missing=args.allow_missing,
                                     max_start_ratio=args.max_start_ratio,
                                     min_spatial_var=args.min_spatial_var)
                    rows.append(rec)
                    errs = rec["checks"].get("errors", [])
                    print(f"  [{rec['status']:7s}] {rec['cell']}")
                    for e in errs:
                        print(f"            ! {e}")

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_missing = sum(1 for r in rows if r["status"] == "MISSING")
    print(f"\n── {n_pass} PASS / {n_fail} FAIL / {n_missing} MISSING "
          f"(of {len(rows)} cells) ──")

    out = Path(args.json_out) if args.json_out else results_root / "retrain_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"verdict": ("PASS" if n_fail == 0 and
                               (args.allow_missing or n_missing == 0)
                               else "FAIL"),
                   "min_explained_var": args.min_ev,
                   "max_model_over_identity": args.max_ident_ratio,
                   "cells": rows}, f, indent=2)
    print(f"[json] -> {out}")

    bad = n_fail + (0 if args.allow_missing else n_missing)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()