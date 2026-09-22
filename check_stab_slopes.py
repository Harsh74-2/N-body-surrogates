#!/usr/bin/env python3
"""check_stab_slopes.py — verify the rollout-slope claims in thesis prose
against results/stability_summary.json:
  1. GNN slope lowest of all 6 variants at every N
  2. stable GNN better slope from N=25 on (exception: N=10 worse)
  3. GNN_st N=100 slope ~16x below GNN sibling and lowest at that count
  4. LSTM N=100 slope jump vs the 1e-4..1e-3 band
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
s = json.loads((REPO / "results" / "stability_summary.json")
               .read_text(encoding="utf-8"))
NS = [10, 25, 50, 100]
V = {"mlp": "mlp_single_step", "mlp_st": "mlp_stable",
     "lstm": "lstm_single_step", "lstm_st": "lstm_stable",
     "gnn": "gnn_single_step", "gnn_st": "gnn_stable"}


def slope(n, v):
    return s[str(n)][V[v]]["mse_slope"]


ok = True
for n in NS:
    vals = {v: slope(n, v) for v in V}
    best = min(vals, key=vals.get)
    print(f"N={n}: " + "  ".join(f"{k}={vals[k]:.2e}" for k in V))
    if best not in ("gnn", "gnn_st"):
        print(f"  !! lowest slope at N={n} is {best}, not a GNN variant")
        ok = False
    if vals["gnn_st"] < vals["gnn"]:
        print(f"  stable GNN better at N={n}: "
              f"{vals['gnn']/vals['gnn_st']:.1f}x")
    else:
        print(f"  stable GNN WORSE at N={n} "
              f"(exception allowed only at N=10)")
        if n != 10:
            ok = False

r = slope(100, "gnn") / slope(100, "gnn_st")
print(f"GNN N=100 sibling/stable ratio: {r:.1f}x (prose says 16x)")
if not (14 <= r <= 18):
    print("  !! ratio drift")
    ok = False

lstm100 = slope(100, "lstm")
print(f"LSTM N=100 single slope: {lstm100:.2e} (prose: jumps to ~4.7e-2)")
if not (1e-2 < lstm100 < 1e-1):
    print("  !! unexpected band")
    ok = False

print("OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)