"""
Parallel mp4 renderer for real-case validation.

Iterates over all (N, preset, model) combos that have `preds.npy` in
`real_case_validation/report_dump/N{n}/preset_*` and renders an mp4
using `real_case_validation.make_animations._animate_one`. Uses
`multiprocessing.Pool` to parallelize across cores (matplotlib is
single-threaded so this is the only way to speed it up).

Writes to `real_case_validation/animations_run/N{n}/preset_*/<model>_<view>.mp4`.

Usage:
    python render_animations_parallel.py             # full 144-mp4 run
    python render_animations_parallel.py --N 50      # just N=50
    python render_animations_parallel.py --preset jupiter_galileans
    python render_animations_parallel.py --frames 200 --fps 15 --view 2d
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from real_case_validation.make_animations import _animate_one


def _render_one(args: tuple) -> tuple:
    """Worker: render one (N, preset, model) => mp4. Returns (combo, status)."""
    n, preset_name, model, view, frames, fps, trail = args
    report_dir = REPO / "real_case_validation" / "report_dump" / f"N{n}"
    out_dir = REPO / "real_case_validation" / "animations_run" / f"N{n}" / f"preset_{preset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Skip only if the mp4 looks COMPLETE: >5 KB AND ends with a moov
    # atom (ffmpeg writes it on the final flush). A partial/interrupted
    # encode of any size is re-rendered.
    label = f"preset_{preset_name}_{model}_{view}"
    expected = out_dir / f"{label}.mp4"
    if expected.exists() and expected.stat().st_size > 5000:
        try:
            with open(expected, "rb") as fh:
                fh.seek(-4096, 2)
                if b"moov" in fh.read():
                    return (f"N{n}/{preset_name}/{model}_{view}",
                            f"SKIP (already rendered)")
        except OSError:
            pass
    t0 = time.time()
    try:
        written = _animate_one(preset_name, model, report_dir, out_dir,
                               view=view, frames=frames, fps=fps, fmt="mp4",
                               trail=trail)
        dt = time.time() - t0
        return (f"N{n}/{preset_name}/{model}_{view}", f"OK {dt:.1f}s -> {[w.name for w in written]}")
    except Exception as e:
        return (f"N{n}/{preset_name}/{model}_{view}", f"FAIL: {e}")


def _discover_jobs(n_list: list | None, preset_filter: str | None,
                   model_filter: str | None) -> list[tuple]:
    """Walk report_dump/N*/preset_*/ and emit jobs for each (model, view)."""
    jobs = []
    dump_root = REPO / "real_case_validation" / "report_dump"
    n_dirs = ([dump_root / f"N{n}" for n in n_list] if n_list
              else sorted(dump_root.glob("N*")))
    for n_dir in n_dirs:
        if not n_dir.is_dir():
            continue
        for p_dir in sorted(n_dir.glob("preset_*")):
            if not (p_dir / "preds.npy").exists():
                continue
            preset_name = p_dir.name.replace("preset_", "")
            if preset_filter and not any(pf in preset_name for pf in preset_filter):
                continue
            import json
            meta_path = p_dir / "preds_meta.json"
            if not meta_path.is_file():
                # Dump dir with preds.npy but no meta (interrupted run)
                # — skip rather than crash discovery.
                print(f"  [skip] {p_dir}: preds.npy without preds_meta.json")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for model in meta["models"]:
                if model_filter and model != model_filter:
                    continue
                jobs.append((int(n_dir.name[1:]), preset_name, model,
                             "3d", 200, 15, 40))
    return jobs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, nargs="*", default=None,
                   help="Specific N values (e.g. 50 100). Default: all.")
    p.add_argument("--preset", action="append", default=None,
                   help="Substring filter on preset name (can be passed multiple "
                        "times to match any of the given substrings)")
    p.add_argument("--model", default=None, help="Specific model name")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--view", choices=("3d", "2d"), default="2d")
    p.add_argument("--trail", type=int, default=40)
    p.add_argument("--workers", type=int, default=6,
                   help="Number of parallel processes (default 6 = physical cores; "
                        "machine has 6P+6HT = 12 logical)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print jobs without running")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    jobs = _discover_jobs(args.N, args.preset, args.model)
    # Override frames/fps/view/trail from CLI
    jobs = [(n, p, m, args.view, args.frames, args.fps, args.trail)
            for (n, p, m, _, _, _, _) in jobs]
    print(f"[plan] {len(jobs)} jobs using {args.workers} workers "
          f"({args.frames} frames @ {args.fps} fps, view={args.view})")
    if args.dry_run:
        for j in jobs:
            print(f"  {j}")
        return 0
    t0 = time.time()
    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_render_one, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                combo, status = fut.result()
                if status.startswith("OK"):
                    done += 1
                    print(f"[ok] {combo}: {status}")
                else:
                    failed += 1
                    print(f"[FAIL] {combo}: {status}", file=sys.stderr)
            except Exception as e:
                failed += 1
                print(f"[FAIL] task crashed: {e}", file=sys.stderr)
    dt = time.time() - t0
    print(f"[done] {done} ok, {failed} failed in {dt:.1f}s "
          f"({dt/max(1, done+failed):.1f}s avg)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
