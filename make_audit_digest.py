#!/usr/bin/env python3
"""make_audit_digest.py — assemble gemini_audit_input.md: every hand-written
numeric/count claim in the thesis, the macro values it must match, and the
verification status, as a self-contained digest for an independent model
cross-check (standing rule: important things cross-verified through a second
model)."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
FORK = REPO / "Universe Simulation Thesis"
M = {}
for line in (FORK / "metrics.tex").read_text(encoding="utf-8").splitlines():
    mo = re.match(r"\\newcommand\{\\(res[A-Za-z]+)\}\{(.*)\}$", line)
    if mo:
        M[mo.group(1)] = mo.group(2)


def v(name):
    s = M[name]
    if s.startswith(">100") or s in ("div.", "--"):
        return None
    s = s.strip("$")
    m = re.match(r"([\d.]+)\\times10\^\{(-?\d+)\}$", s)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    return float(s.replace("\\%", ""))


NS = ["Nten", "Ntwentyfive", "Nfifty", "Nhundred"]
NSP = {"Nten": 10, "Ntwentyfive": 25, "Nfifty": 50, "Nhundred": 100}
L: list[str] = []
L.append("# Thesis audit digest (2026-09-23)\n")
L.append("Purpose: independent cross-check of every hand-written numeric "
         "claim in the thesis against the generated macro values "
         "(metrics.tex, 1520 macros generated from the final retrain "
         "artifacts). Claims quoted verbatim from thesis.tex / "
         "thesis_appendix_results.tex; the value lines are the macro "
         "contents.\n")

# 1. OOD survivor means
L.append("## Claim 1 — OOD survivor means (tab:ood) / tab:stable-vs-single")
L.append("Thesis: 'the stable variant lands within five points of its "
         "sibling in 10 of the 11 comparable cells'; 'stability-trained GNN "
         "degrades from X to Y at N=100, while the stability-trained LSTM "
         "is the only variant with any surviving OOD preset there at all'.")
L.append("Values (survivor mean %, single -> stable, delta in pp):")
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        s, st = v(f"resRollOOD{m}{n}"), v(f"resRollOOD{m}St{n}")
        d = "n/a" if (s is None or st is None) else f"{st - s:+.1f}"
        L.append(f"  {m} {NSP[n]:>3}: single={s}  stable={st}  delta={d}")
L.append("")

# 2. 8 of 12 single-step
L.append("## Claim 2 — 'stable beats sibling on single-step MSE in 8 of 12 pairs'")
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        a, b = v(f"res{m}Mse{n}"), v(f"res{m}StMse{n}")
        L.append(f"  {m} {NSP[n]:>3}: single={a:.3g}  stable={b:.3g}  "
                 f"stable-wins={b < a}")
L.append("")

# 3. slope 10 of 12
L.append("## Claim 3 — 'improves the K=128 rollout slope in 10 of 12 (two "
         "exceptions: GNN at N=10 and a 0.06% tie at the LSTM's N=50)'")
ss = json.loads((REPO / "results" / "stability_summary.json")
                .read_text(encoding="utf-8"))
V = {"mlp": ("Mlp", "mlp_single_step"), "mlp_st": ("MlpSt", "mlp_stable"),
     "lstm": ("Lstm", "lstm_single_step"), "lstm_st": ("LstmSt", "lstm_stable"),
     "gnn": ("Gnn", "gnn_single_step"), "gnn_st": ("GnnSt", "gnn_stable")}
for n in NSP:
    for a, b in (("mlp", "mlp_st"), ("lstm", "lstm_st"), ("gnn", "gnn_st")):
        x = ss[str(NSP[n])][V[a][1]]["mse_slope"]
        y = ss[str(NSP[n])][V[b][1]]["mse_slope"]
        L.append(f"  {V[a][0]} {NSP[n]:>3}: single={x:.4e}  stable={y:.4e}  "
                 f"stable-better={y < x}")
lstm50a = ss["50"]["lstm_single_step"]["mse_slope"]
lstm50b = ss["50"]["lstm_stable"]["mse_slope"]
L.append(f"  LSTM N=50 relative gap: {abs(lstm50b - lstm50a) / lstm50a * 100:.3f}% "
         f"(thesis says 0.06% tie)")
L.append("")

# 4. GNN slope lowest + 16x
L.append("## Claim 4 — 'GNN lowest rollout slope at every N, one to two "
         "orders of magnitude below MLP/LSTM'; 'stable GNN better from "
         "N=25 on'; 'GNN_st N=100 slope ~16x below sibling'")
for n in NSP:
    vals = {k: ss[str(NSP[n])][vv[1]]["mse_slope"] for k, vv in V.items()}
    best = min(vals, key=vals.get)
    L.append(f"  N={NSP[n]:>3}: best={best} ({vals[best]:.3e})  "
             f"GNN={vals['gnn']:.3e}  GNN_st={vals['gnn_st']:.3e}  "
             f"MLP={vals['mlp']:.3e}  LSTM={vals['lstm']:.3e}")
r = ss["100"]["gnn_single_step"]["mse_slope"] / ss["100"]["gnn_stable"]["mse_slope"]
L.append(f"  GNN N=100 sibling/stable slope ratio: {r:.1f}x")
L.append("")

# 5. GNN 1e-8 band wording
L.append("## Claim 5 — GNN 1e-8 single-step band wording")
L.append("Thesis (interpretation): 'at N_train=50 it reaches the 1e-8 "
         "single-step band of the MLP (resGnnMseNfifty, the best GNN cell "
         "of the sweep), with its stability-trained sibling holding that "
         "band at N=100 (resGnnStMseNhundred)'.")
L.append("(Note the claim is deliberately NOT made for the single-step GNN "
         "at N=100.)")
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        L.append(f"  {m} {NSP[n]:>3}: {v(f'res{m}Mse{n}'):.3g}")
    for n in NS:
        L.append(f"  {m}St {NSP[n]:>3}: {v(f'res{m}StMse{n}'):.3g}")
L.append("")

# 6. disc / OOD single-step bands
L.append("## Claim 6 — single-step error bands (sec:res:ood)")
disc = [v(f"resSs{m}Disc{n}") for m in ("Mlp", "Lstm", "Gnn") for n in NS]
ood = [v(f"resSs{m}{p}{n}") for m in ("Mlp", "Lstm", "Gnn")
       for p in ("Fss", "Ip", "Jg", "Sse", "Seo", "Spm") for n in NS]
L.append(f"Thesis: 'single-step mean error on the disc is 0.00-0.03% of L "
         f"for all three architectures at every count, against 0.3-29.9% "
         f"on the OOD presets (inner_planets the high end, "
         f"solar_system_extended the low end)'")
L.append(f"Computed: disc range {min(disc):.2f}-{max(disc):.2f};  "
         f"OOD range {min(ood):.2f}-{max(ood):.2f}")
L.append(f"  disc min cell:  {min(disc)}")
L.append(f"  OOD max = inner_planets? "
         f"{max((v(f'resSs{m}Ip{n}'), 'Ip', m, n) for m in ('Mlp','Lstm','Gnn') for n in NS)}")
L.append(f"  OOD min = solar_system_extended? "
         f"{min((v(f'resSs{m}Sse{n}'), 'Sse', m, n) for m in ('Mlp','Lstm','Gnn') for n in NS)}")
L.append("")

# 7. jupiter_galileans band
L.append("## Claim 7 — jupiter_galileans no-compounding band "
         "'all three architectures sit in a ~10.7-11.4% band across N'")
jg = [v(f"resSs{m}Jg{n}") for m in ("Mlp", "Lstm", "Gnn") for n in NS]
L.append(f"Computed jg range: {min(jg):.2f}-{max(jg):.2f}  "
         f"values: {sorted(set(round(x, 2) for x in jg))}")
L.append("")
L.append("## Claim 8 — latency appendix")
lat = json.loads((REPO / "results" / "latency_bench.json")
                 .read_text(encoding="utf-8"))
L.append("Thesis: 'solver O(N^2) from 0.029 ms at N=10 to 1.62 ms at N=200; "
         "MLP crossover only at N*=200 single-frame / N*=100 batched; LSTM "
         "above the solver at every count; GNN slowest surrogate at every N'")
for k in ("10", "25", "50", "100", "200"):
    L.append(f"  solver N={k}: {lat['solver'][k]:.4g} ms")
for mode in ("surrogate_single_frame", "surrogate_batched_amortised"):
    for a in ("mlp", "lstm", "gnn"):
        row = lat[mode][a]
        L.append(f"  {a} {mode}: "
                 + "  ".join(f"N{k}={row[k]:.4g}" for k in
                             ("10", "25", "50", "100", "200")))
L.append("")

out = REPO / "gemini_audit_input.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size:,} bytes)")