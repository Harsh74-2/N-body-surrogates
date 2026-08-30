"""
One-off: render disc_imf_in_distribution_baseline mp4 animations for all 4
training N. The runner (real_case_runner.py) writes
  report_N{n}/preset_disc_imf_in_distribution_baseline/{preds.npy,
  book_pos.npy, preds_meta.json}
under --out, and make_animations.py reads them. The book_pos.npy slice is
aligned to preds.npy (both length n_rollout_ref = T-W) so the broadcast in
make_animations.py:611 matches.

For each N, runs:
  1. the runner for the dist preset with --dump-preds
  2. make_animations.py with --preset disc_imf_in_distribution_baseline
     --report-dir real_case_validation/report_N{n}

Animations go into real_case_validation/animations/ with the canonical
N{n}_{preset}_{human}.mp4 naming (set by render_animations_parallel.py's
output convention; make_animations.py writes <out>/preset_<NAME>_<MODEL>_<VIEW>.mp4
which we rename after).

Usage:
    python -m real_case_validation._render_dist_anims
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NS = [10, 25, 50, 100]
DIST_PRESET = "disc_imf_in_distribution_baseline"

VARIANTS = [
    ("gnn",  "gnn",  "GNN"),
    ("gnn_stable",  "gnn",  "GNN_stable"),
    ("lstm", "lstm", "LSTM"),
    ("lstm_stable", "lstm", "LSTM_stable"),
    ("mlp",  "mlp",  "MLP"),
    ("mlp_stable",  "mlp",  "MLP_stable"),
]

ANIM_OUT_DIR = REPO_ROOT / "real_case_validation" / "animations"


def _run_runner(n: int) -> bool:
    """Re-run the runner for the dist preset at this N with --dump-preds.
    Returns True on success."""
    out_dir = REPO_ROOT / "real_case_validation" / f"report_N{n}"
    preset_dir = out_dir / f"preset_{DIST_PRESET}"
    preset_dir.mkdir(parents=True, exist_ok=True)

    # Wipe stale preds/book_pos/meta so a fresh dump supersedes any prior.
    for fname in ("preds.npy", "book_pos.npy", "preds_meta.json"):
        f = preset_dir / fname
        if f.exists():
            f.unlink()

    ckpt_args = []
    for sub, surr, human in VARIANTS:
        ckpt = REPO_ROOT / "training_runs" / f"N{n}" / sub / "model_best.pt"
        if not ckpt.is_file():
            print(f"  [skip] {ckpt}", flush=True)
            continue
        ckpt_args += ["--ckpt",
                      f"{str(ckpt.relative_to(REPO_ROOT)).replace(chr(92), '/')}"
                      f":{surr}:{human}"]
    if not ckpt_args:
        print(f"[N={n}] no ckpts present", flush=True)
        return False

    cmd = [sys.executable, "-m",
           "real_case_validation.real_case_runner",
           "--preset", DIST_PRESET,
           "--out", str(out_dir),
           "--quick",
           "--dump-preds"] + ckpt_args
    print(f"\n[runner N={n}] preset={DIST_PRESET}  ckpts={len(ckpt_args)}",
          flush=True)
    try:
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=1200)
        if r.returncode != 0:
            print(f"  [fail] exit {r.returncode}", flush=True)
            print(r.stderr[-2000:], flush=True)
            return False
    except subprocess.TimeoutExpired:
        print("  [timeout 1200s]", flush=True)
        return False

    # Sanity-check the dumped arrays.
    import numpy as np
    preds = np.load(preset_dir / "preds.npy")
    book = np.load(preset_dir / "book_pos.npy")
    print(f"  [ok] preds={preds.shape}  book={book.shape}", flush=True)
    if preds.shape[1] != book.shape[0]:
        print(f"  [SHAPE MISMATCH] preds T={preds.shape[1]} != book T={book.shape[0]}",
              flush=True)
        return False
    return True


def _render_animations(n: int) -> int:
    """Render mp4s for the dist preset at this N. Returns count of mp4s
    successfully renamed into ANIM_OUT_DIR with the canonical N{n}_* name."""
    out_dir = REPO_ROOT / "real_case_validation" / f"report_N{n}"
    preset_dir = out_dir / f"preset_{DIST_PRESET}"
    if not (preset_dir / "preds.npy").is_file():
        print(f"[render N={n}] no preds.npy under {preset_dir}", flush=True)
        return 0

    # make_animations.py writes to args.out (default real_case_validation/animations/)
    # and names files <out>/preset_<NAME>_<MODEL>_<VIEW>.mp4. We rename to
    # canonical N{n}_<NAME>_<MODEL>.mp4 to match the rest of the gallery.
    #
    # Cap at --frames 200 to match the existing 144-mp4 gallery (the
    # `render_animations_parallel.py` convention). Uncapped = 2495 frames
    # per clip × 25 bodies is hours of matplotlib render per mp4 and
    # not what the gallery uses.
    ANIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m",
           "real_case_validation.make_animations",
           "--preset", DIST_PRESET,
           "--report-dir", str(out_dir),
           "--view", "2d",
           "--out", str(ANIM_OUT_DIR),
           "--format", "mp4",
           "--fps", "30",
           "--frames", "200"]
    print(f"\n[render N={n}] {DIST_PRESET} x 6 models x 2d", flush=True)
    try:
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=7200)
        if r.returncode != 0:
            print(f"  [fail] exit {r.returncode}", flush=True)
            print(r.stderr[-2000:], flush=True)
            return 0
    except subprocess.TimeoutExpired:
        print("  [timeout 7200s]", flush=True)
        return 0

    renamed = 0
    for fname in ANIM_OUT_DIR.glob(f"preset_{DIST_PRESET}_*_2d.mp4"):
        # preset_<NAME>_<MODEL>_2d.mp4 -> N{n}_<NAME>_<MODEL>.mp4
        model_part = fname.stem.replace(f"preset_{DIST_PRESET}_", "").rsplit("_2d", 1)[0]
        target = ANIM_OUT_DIR / f"N{n}_{DIST_PRESET}_{model_part}.mp4"
        if target.exists():
            target.unlink()
        shutil.move(str(fname), str(target))
        renamed += 1
    print(f"  [ok] {renamed} mp4s renamed", flush=True)
    return renamed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-runner", action="store_true",
                    help="Only re-run the runner (skip mp4 render).")
    ap.add_argument("--only-render", action="store_true",
                    help="Only render mp4s (assume runner output exists).")
    args = ap.parse_args()

    total_renamed = 0
    for n in NS:
        if not args.only_render:
            ok = _run_runner(n)
            if not ok:
                continue
        if args.only_runner:
            continue
        total_renamed += _render_animations(n)

    print(f"\nDone. mp4s rendered+renamed: {total_renamed}")


if __name__ == "__main__":
    main()
