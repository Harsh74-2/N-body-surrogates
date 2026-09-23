#!/usr/bin/env python3
"""verify_thesis_claims.py — recompute every hand-written count/delta claim
in thesis.tex + thesis_appendix_results.tex from the generated macro file."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
errs: list[str] = []
FORK = REPO / "Universe Simulation Thesis"
txt = (FORK / "metrics.tex").read_text(encoding="utf-8")
M = {}
for line in txt.splitlines():
    mo = re.match(r'\\newcommand\{\\(res[A-Za-z]+)\}\{(.*)\}$', line)
    if mo:
        M[mo.group(1)] = mo.group(2)


def v(name):
    s = M[name]
    if s.startswith(">100") or s in ("div.", "--"):
        return None
    s = s.strip("$")
    m = re.match(r'([\d.]+)\\times10\^\{(-?\d+)\}$', s)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    return float(s.replace("\\%", ""))


NS = ["Nten", "Ntwentyfive", "Nfifty", "Nhundred"]

print("== tab:stable-vs-single OOD survivor mean deltas (stable - single, pp) ==")
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        s, st = v(f"resRollOOD{m}{n}"), v(f"resRollOOD{m}St{n}")
        d = "n/a" if (s is None or st is None) else f"{st - s:+.1f}"
        print(f"  {m:4s} {n:12s}: single={s} stable={st} delta={d}")

print("\n== 'stable beats sibling in 8 of 12' (in-dist single-step MSE, tab:eval_full) ==")
win = []
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        a, b = v(f"res{m}Mse{n}"), v(f"res{m}StMse{n}")
        w = b < a
        win.append(w)
        print(f"  {m:4s} {n:12s}: single={a:.3g} stable={b:.3g} -> stable wins: {w}")
print(f"  => stable wins {sum(win)}/12")

print("\n== 'within a few points ... 10 of 11 comparable cells' (OOD survivor mean) ==")
comp, within = 0, 0
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        s, st = v(f"resRollOOD{m}{n}"), v(f"resRollOOD{m}St{n}")
        if s is None or st is None:
            print(f"  {m:4s} {n:12s}: NOT comparable (single={s}, stable={st})")
            continue
        comp += 1
        d = abs(st - s)
        w = d <= 5.0
        within += w
        print(f"  {m:4s} {n:12s}: |delta|={d:.1f} within-5pp={w}")
print(f"  => comparable {comp}, within 5pp {within}")

print("\n== OOD band claim: 21 of 23 comparable cells in [63.6, 74.3] ==")
vals, inband = [], []
for m in ("Mlp", "MlpSt", "Lstm", "LstmSt", "Gnn", "GnnSt"):
    for n in NS:
        x = v(f"resRollOOD{m}{n}")
        if x is None:
            continue
        vals.append((m, n, x))
        inband.append(63.6 <= x <= 74.3)
outs = [(m, n, x) for (m, n, x), b in zip(vals, inband) if not b]
print(f"  numeric cells {len(vals)}, in band {sum(inband)}, outside: {outs}")

TOKS = ("Mlp", "MlpSt", "Lstm", "LstmSt", "Gnn", "GnnSt")
print("\n== 'GNN rollout slope lowest of all 24 cells at every N' (tab:stability) ==")
for n in NS:
    vals = {k: v(f"resStab{k}MseSlope{n}") for k in TOKS}
    best = min(vals, key=vals.get)
    print(f"  {n:12s}: best={best} ({vals[best]:.3g}) "
          f"GNN={vals['Gnn']:.3g} GNN_st={vals['GnnSt']:.3g}")
print("  MLP/LSTM N10 slopes ~1e-3?",
      f"MLP={v('resStabMlpMseSlopeNten'):.3g}",
      f"LSTM={v('resStabLstmMseSlopeNten'):.3g}")

print("\n== 'stability-trained GNN better from N=25 on' (slope, tab:stability) ==")
for n in NS:
    a = v(f"resStabGnnMseSlope{n}")
    b = v(f"resStabGnnStMseSlope{n}")
    print(f"  {n:12s}: single={a:.3g} stable={b:.3g} stable-better={b < a}")

print("\n== in-dist single-step MSE ranking (tab:eval_full context) ==")
cells = sorted(((m, n, v(f"res{m}Mse{n}")) for m in ("Mlp", "Lstm", "Gnn")
                for n in NS), key=lambda c: c[2])
for m, n, x in cells:
    print(f"  {m:4s} {n:12s}: {x:.3g}")
print("  best:", cells[0])

print("\n== '$10^{-8}$ band' claims (thesis: MLP at N=10/25/100, LSTM at N=25/100,")
print("   GNN single at N=50, GNN stable at N=100; band = [1e-8, 1e-7) loosely) ==")
BAND = lambda x: 1e-8 <= x < 1e-7
band_claims = [
    ("MLP N=10",  v("resMlpMseNten"),  True),
    ("MLP N=25",  v("resMlpMseNtwentyfive"), True),
    ("MLP N=50",  v("resMlpMseNfifty"), False),   # 1.2e-07, prose excludes it
    ("MLP N=100", v("resMlpMseNhundred"), True),
    ("LSTM N=25", v("resLstmMseNtwentyfive"), True),
    ("LSTM N=100", v("resLstmMseNhundred"), True),
    ("GNN N=50 (single)", v("resGnnMseNfifty"), True),
    ("GNN-stable N=100", v("resGnnStMseNhundred"), True),
]
for label, x, want in band_claims:
    got = BAND(x)
    print(f"  {label:20s}: {x:.3g} in-band={got} (prose expects {want})")
    if got != want:
        errs.append(f"band claim {label}: {x:.3g} in-band={got} but prose expects {want}")

print("\n== 'LSTM most accurate single-step surrogate at N=25' ==")
for n in ("Ntwentyfive",):
    vals = {m: v(f"res{m}Mse{n}") for m in ("Mlp", "Lstm", "Gnn")}
    print(" ", vals, "-> best:", min(vals, key=vals.get))

print("\n== '0.00-0.03% disc ss' and '0.3-29.9% OOD ss' (thesis 1448) ==")
# read from the 2-decimal source JSONs; the 1-decimal macros cannot show 0.03
ss_vals: dict[str, list[float]] = {"disc": [], "ood": []}
for n in (10, 25, 50, 100):
    for pd in (REPO / "real_case_validation" / f"report_N{n}" / "single_step").glob("preset_*"):
        j = json.loads((pd / "ss_summary.json").read_text(encoding="utf-8"))
        lane = "disc" if j.get("in_distribution") else "ood"
        for mv in j["per_model"].values():
            ss_vals[lane].append(mv["mean_err_pct"])
dmin, dmax = min(ss_vals["disc"]), max(ss_vals["disc"])
omin, omax = min(ss_vals["ood"]), max(ss_vals["ood"])
print(f"  disc range: {dmin:.4f}-{dmax:.4f}  (prose: 0.00-0.03)")
print(f"  OOD ss range: {omin:.2f}-{omax:.2f}  (prose: 0.3-29.9)")
if not (0.0 <= dmin and dmax <= 0.03):
    errs.append(f"disc ss range {dmin:.4f}-{dmax:.4f} outside prose 0.00-0.03")
if not (0.25 <= omin and omax <= 30.0):
    errs.append(f"OOD ss range {omin:.2f}-{omax:.2f} outside prose 0.3-29.9")

print()
if errs:
    print(f"MISMATCHES ({len(errs)}):")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print("ALL RECOMPUTED CLAIMS AGREE WITH THE THESIS PROSE")