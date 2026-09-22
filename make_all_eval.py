#!/usr/bin/env python3
"""
make_all_eval.py — assemble results/all_eval.json from the per-cell
evaluate_models.py outputs (nothing hand-transcribed).

Input metrics files (both written by evaluate_models.py, --json):
  single-step : results/N{n}/metrics_{m}.json          (scaling_sweep.py)
  stable      : results/N{n}_{m}_stable_metrics.json   (train_stable_variants.sh,
                written at the results root)

Output: results/all_eval.json — 24 records (4 N x 6 variants), in the
exact pop(0) order make_site_audits.load_all_eval consumes:
for N in (10, 25, 50, 100): for model in (mlp, lstm, gnn): single, stable.

Each metrics json is a list of Metrics dataclass records; the test-split
record is used (every cell here is a test-split evaluation, but the filter
guards against future multi-split files).

Usage:  python make_all_eval.py            (run from the repo root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

N_VALUES = [10, 25, 50, 100]
FAMILY = ["mlp", "lstm", "gnn"]


def test_record(path: Path) -> dict:
    recs = json.loads(path.read_text(encoding="utf-8"))
    tests = [r for r in recs if r.get("split") == "test"] or recs
    return tests[0]


def main() -> int:
    root = Path(".")
    out: list[dict] = []
    missing: list[str] = []
    for n in N_VALUES:
        for m in FAMILY:
            for label, p in (
                (f"N{n}/{m} (single)", root / f"results/N{n}/metrics_{m}.json"),
                (f"N{n}/{m} (stable)",
                 root / f"results/N{n}_{m}_stable_metrics.json"),
            ):
                if not p.is_file():
                    missing.append(f"{label}: {p}")
                    continue
                rec = test_record(p)
                rec.setdefault("variant", "stable" if "(stable)" in label
                               else "single_step")
                rec["cell"] = f"N{n}/{m}"
                out.append(rec)

    if missing:
        print("MISSING metrics files — run the eval step for these cells "
              "first (scaling_sweep step 6/6 for single-step cells, "
              "train_stable_variants.sh for stable cells):")
        for s in missing:
            print(f"  - {s}")
        return 1

    if len(out) != 24:
        print(f"expected 24 records, assembled {len(out)} — aborting")
        return 1

    dst = root / "results" / "all_eval.json"
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[json] -> {dst}  (24 cells, order: N asc, mlp/lstm/gnn, "
          f"single before stable)")

    # Print the two cells make_site_audits uses as ordering anchors, so
    # its constants can be cross-checked against the fresh data.
    for n in (25, 100):
        idx = N_VALUES.index(n) * 6 + 4          # gnn single-step slot
        print(f"anchor gnn N={n} single-step mse = {out[idx]['mse']:.4e}  "
              f"(rollout {out[idx]['rollout']:.3e}, "
              f"latency {out[idx]['latency_ms']:.3f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())