#!/usr/bin/env python3
"""
rebuild_site.py — one-command site rebuild + deploy for the post-retrain round.

Run this LOCALLY after the VM artifacts are synced into this clone
(git pull). It chains the full Phase-C pipeline; every step fails hard
(nonzero exit stops the chain) so a broken stage never reaches the live
site, and every step is idempotent so a partial failure is fixed by
re-running the same command.

  1. make_site_audits.py            cross_N_audit mds + hub fragment
  2. regen_top_level_plots.py       eval_benchmark/stability/scaling_latency PNGs
  3. build_interactive_animations.py 336 HTML viewers from the fresh dumps
  4. build_github_pages.py          rebuild gh_pages_build/
  5. deploy_github_pages.py         push the `pages` branch → live

Usage:
    python rebuild_site.py                 # full chain + deploy
    python rebuild_site.py --no-deploy     # build + verify, don't push
    python rebuild_site.py --from 3        # skip steps 1-2 (idempotent rerun)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

STEPS: list[tuple[int, str, list[str]]] = [
    (1, "make_site_audits.py", [sys.executable, "make_site_audits.py"]),
    (2, "regen_top_level_plots.py",
     [sys.executable, "regen_top_level_plots.py",
      "--cross-n-single-step-md", "results/cross_N_audit_single_step.md",
      "--cross-n-rollout-md", "results/cross_N_audit.md",
      "--latency-json", "results/latency_bench.json",
      "--out", "plots"]),
    (3, "build_interactive_animations.py",
     [sys.executable, "build_interactive_animations.py"]),
    (4, "build_github_pages.py", [sys.executable, "build_github_pages.py"]),
    (5, "deploy_github_pages.py", [sys.executable, "deploy_github_pages.py"]),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-deploy", action="store_true",
                   help="Run steps 1-4 only (verify without pushing).")
    p.add_argument("--from", type=int, default=1, dest="from_step",
                   help="First step to run (1-5); earlier steps are skipped.")
    args = p.parse_args()

    last = 4 if args.no_deploy else 5
    for num, name, cmd in STEPS:
        if num < args.from_step or num > last:
            print(f"[skip] step {num}: {name}")
            continue
        print(f"[step {num}] {name} ...", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[FAIL] step {num} ({name}) exited {rc} — chain stopped "
                  f"before the live site could see a half-built state.")
            return rc
    print("[done] site rebuilt" + ("" if args.no_deploy else " and deployed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())