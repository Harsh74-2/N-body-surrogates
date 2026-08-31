#!/usr/bin/env python3
"""
run_animations.py
=================
End-to-end pipeline for the rollout animations:

  for N in {10, 25, 50, 100}:
    for preset in PRESETS:
      for variant in {mlp, mlp_stable, lstm, lstm_stable, gnn, gnn_stable}:
        1. call real_case_runner with --ckpt <variant> --dump-preds
        2. make_animations.py reads preds.npy -> *.mp4

Inference-only (no retraining), CPU-compatible, ~hours of wall clock.

Skips presets that are obviously OOD-heavy and long (e.g. full_solar_system
when the per-N report already covers them) to keep this script runnable in
a single session. Pass --all to disable that filter.

Fast re-render path: `render_animations_parallel.py` skips the
`real_case_runner` step and reads `preds.npy` already on disk under
`real_case_validation/report_dump/`. Use it whenever the inference
artefacts exist and you only want to (re)draw the animations.

Usage:
    python run_animations.py                    # default N={50,100}, all presets
    python run_animations.py --n 50 100         # explicit N
    python run_animations.py --presets jupiter_galileans sun_earth_only
    python run_animations.py --variants lstm gnn_stable
    python run_animations.py --all              # all 4 N, all 7 presets, all 6 variants
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Map (variant_name) -> (training_runs subdir, surrogate_type, human_name)
VARIANTS = {
    "mlp":         ("mlp",         "mlp",  "MLP"),
    "mlp_stable":  ("mlp_stable",  "mlp",  "MLP_stable"),
    "lstm":        ("lstm",        "lstm", "LSTM"),
    "lstm_stable": ("lstm_stable", "lstm", "LSTM_stable"),
    "gnn":         ("gnn",         "gnn",  "GNN"),
    "gnn_stable":  ("gnn_stable",  "gnn",  "GNN_stable"),
}

# 7 presets, ordered from cheapest (top) to most-expensive (bottom).
PRESETS_ORDERED = [
    "sun_earth_only",            # 2 bodies, fastest
    "jupiter_galileans",         # 5 bodies (Jupiter + 4 Galilean moons)
    "moon",                      # alias sun_planets_moon
    "inner_planets",             # 5 bodies
    "outer",                     # alias full_solar_system
    "extended",                  # alias solar_system_extended
    "dist",                      # alias disc_imf_in_distribution_baseline
]

# Map short aliases accepted by run_animations.py to the canonical
# preset names that the runner actually understands. The runner does
# not do alias resolution, so we resolve here before invoking it.
PRESET_ALIASES = {
    "moon":     "sun_planets_moon",
    "outer":    "full_solar_system",
    "extended": "solar_system_extended",
    "dist":     "disc_imf_in_distribution_baseline",
}


def _canonical_preset(name: str) -> str:
    return PRESET_ALIASES.get(name, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--presets", nargs="+", default=None,
                    help="Preset names or filter aliases; default = "
                         "all 7 in PRESETS_ORDERED.")
    ap.add_argument("--variants", nargs="+",
                    default=list(VARIANTS.keys()),
                    help="Variant keys from VARIANTS; default = all 6.")
    ap.add_argument("--out-root", default="real_case_validation/animations_run",
                    help="Per-N output root for preds + mp4.")
    ap.add_argument("--report-root", default=None,
                    help="Use existing per-N report dir instead of a "
                         "fresh --dump-preds run (preds must already "
                         "exist). Skips the runner step.")
    ap.add_argument("--frames", type=int, default=800,
                    help="Max frames per mp4 (full trajectory if smaller).")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--all", action="store_true",
                    help="Equivalent to --n 10 25 50 100 --presets all.")
    args = ap.parse_args()

    if args.all:
        args.n = [10, 25, 50, 100]
        args.presets = PRESETS_ORDERED

    presets = args.presets or PRESETS_ORDERED

    n_total = len(args.n) * len(presets)
    print(f"Plan: {len(args.n)} N x {len(presets)} presets "
          f"({len(args.variants)} variants each) = "
          f"{n_total} (run, {len(args.variants)} mp4s) batches",
          flush=True)

    failures: list[tuple[int, str, str, str]] = []

    for n in args.n:
        for preset in presets:
            # Canonicalize short aliases (moon/outer/extended/dist) once,
            # up front: the runner, the preds dir, and make_animations all
            # key on the canonical preset name — resolving only at the
            # runner call would desync the three.
            preset = _canonical_preset(preset)
            # Build one runner invocation that loads every requested
            # variant at once. The runner writes one preds.npy with
            # all variants stacked along axis 0 — that is the only
            # way to get a multi-variant animation source.
            ckpts: list[str] = []
            subdirs: dict[str, str] = {}  # human -> subdir for fallback
            for var_key in args.variants:
                subdir, surr_type, human = VARIANTS[var_key]
                ckpt = REPO_ROOT / "training_runs" / f"N{n}" / subdir / "model_best.pt"
                if not ckpt.is_file():
                    print(f"  [skip] ckpt missing: {ckpt}", flush=True)
                    continue
                ckpt_str = str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/")
                ckpts.append(f"{ckpt_str}:{surr_type}:{human}")
                subdirs[human] = subdir
            if not ckpts:
                continue

            # 1) --dump-preds run on this (N, preset) — one runner
            #    invocation that loads every requested ckpt and writes
            #    a single preds.npy with all variants stacked along axis 0.
            if args.report_root:
                # The runner writes preds to `<out_root>/N{n}/preset_<name>/`.
                # Per-N `report_N{n}/preset_<name>/` is an alternate layout
                # produced by old audits. Try both; whichever has
                # preds.npy wins.
                candidates = [
                    Path(args.report_root) / f"report_N{n}" / f"preset_{preset}",
                    Path(args.report_root) / f"N{n}" / f"preset_{preset}",
                ]
                preset_dir = next(
                    (c for c in candidates if (c / "preds.npy").exists()),
                    None,
                )
                if preset_dir is None:
                    print(f"  [skip] no preds.npy in any of "
                          f"{[str(c) for c in candidates]}",
                          flush=True)
                    continue
            else:
                run_out = REPO_ROOT / args.out_root / f"N{n}"
                preset_dir = run_out / f"preset_{preset}"
                preset_dir.mkdir(parents=True, exist_ok=True)
                if (preset_dir / "preds.npy").exists():
                    print(f"  [reuse] {preset_dir}/preds.npy",
                          flush=True)
                else:
                    cmd = [sys.executable, "-m",
                           "real_case_validation.real_case_runner",
                           "--preset", preset,
                           "--out", str(run_out),
                           "--dump-preds",
                           "--quick"]
                    # Insert every --ckpt positional arg.
                    for c in ckpts:
                        cmd[3:3] = ["--ckpt", c]   # noqa: E501
                    print(f"  [run] N={n} preset={preset} "
                          f"({len(ckpts)} ckpts)", flush=True)
                    try:
                        r = subprocess.run(cmd, cwd=REPO_ROOT,
                                           capture_output=True, text=True,
                                           timeout=1800)
                        if r.returncode != 0:
                            print(f"    [fail] exit {r.returncode}",
                                  flush=True)
                            print(r.stderr[-1500:], flush=True)
                            failures.append((n, preset, ",".join(subdirs),
                                             f"runner exit {r.returncode}"))
                            continue
                    except subprocess.TimeoutExpired:
                        print("    [timeout 1800s]", flush=True)
                        failures.append((n, preset, ",".join(subdirs),
                                         "timeout"))
                        continue

            # 2) animation render — one mp4 per requested variant.
            anim_dir = REPO_ROOT / "real_case_validation" / "animations"
            anim_dir.mkdir(parents=True, exist_ok=True)
            for var_key in args.variants:
                _, _, human = VARIANTS[var_key]
                if human not in subdirs:
                    continue  # skipped above (ckpt missing)
                out_mp4 = anim_dir / f"N{n}_{preset}_{human}.mp4"
                if out_mp4.exists():
                    print(f"  [reuse] {out_mp4.name}", flush=True)
                    continue
                # The make_animations.py script writes files of the form
                # `preset_<preset>_<model>_<view>.{mp4,gif}`. We rename
                # its output to the canonical `N{n}_{preset}_{human}.mp4`
                # so the orchestrator's idempotent skip works correctly.
                produced_mp4 = (anim_dir
                               / f"preset_{preset}_{human}_3d.mp4")
                if produced_mp4.exists():
                    print(f"  [reuse] {out_mp4.name} (rename from "
                          f"{produced_mp4.name})", flush=True)
                    produced_mp4.rename(out_mp4)
                    continue
                cmd = [
                    sys.executable, "-m",
                    "real_case_validation.make_animations",
                    "--report-dir", str(preset_dir.parent),
                    "--preset", preset,
                    "--model", human,
                    "--out", str(anim_dir),
                    "--frames", str(args.frames),
                    "--fps", str(args.fps),
                    "--format", "mp4",
                ]
                print(f"  [anim] {out_mp4.name}", flush=True)
                try:
                    r = subprocess.run(cmd, cwd=REPO_ROOT,
                                       capture_output=True, text=True,
                                       timeout=1800)
                    if r.returncode != 0:
                        print(f"    [fail] exit {r.returncode}",
                              flush=True)
                        print(r.stderr[-1500:], flush=True)
                        failures.append((n, preset, var_key,
                                         f"anim exit {r.returncode}"))
                    elif produced_mp4.exists():
                        produced_mp4.rename(out_mp4)
                except subprocess.TimeoutExpired:
                    print("    [timeout 1800s]", flush=True)
                    failures.append((n, preset, var_key, "anim timeout"))

    print(f"\nDone. Failures: {len(failures)}")
    for n, preset, var_key, why in failures:
        print(f"  N={n} preset={preset} variant={var_key}: {why}")


if __name__ == "__main__":
    main()