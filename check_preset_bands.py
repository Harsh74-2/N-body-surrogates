#!/usr/bin/env python3
import re
from pathlib import Path

FORK = Path(__file__).resolve().parent / "Universe Simulation Thesis"
txt = (FORK / "metrics.tex").read_text(encoding="utf-8")
M = {}
for line in txt.splitlines():
    mo = re.match(r'\\newcommand\{\\(res[A-Za-z]+)\}\{(.*)\}$', line)
    if mo:
        M[mo.group(1)] = mo.group(2)


def num(name):
    s = M[name].strip("$")
    m = re.match(r'([\d.]+)\\times10\^\{(-?\d+)\}$', s)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    if M[name] == "div.":
        return None
    return float(s.replace("\\%", ""))


NS = ["Nten", "Ntwentyfive", "Nfifty", "Nhundred"]

print("== per-preset single-step values ==")
for p in ("Sse", "Seo", "Spm"):
    print(f"  {p}:")
    for m in ("Mlp", "Lstm", "Gnn"):
        print(f"    {m:4s}", [num(f"resSs{m}{p}{n}") for n in NS])

print("\n== rollout Ip + Seo grid (6 variants x 4N, calibrated mean) ==")
for p in ("Sse",):
    print(f"  {p}:")
    for m in ("Mlp", "MlpSt", "Lstm", "LstmSt", "Gnn", "GnnSt"):
        print(f"    {m:6s}", [num(f"resRoll{m}{p}{n}Mean") for n in NS])

print("\n== rollout Ip/Seo ranges claimed: Ip 63-74, Seo 64-92 ==")
for p in ("Sse",):
    vals = [num(f"resRoll{m}{p}{n}Mean") for m in
            ("Mlp", "MlpSt", "Lstm", "LstmSt", "Gnn", "GnnSt") for n in NS]
    vals = [x for x in vals if x is not None]
    print(f"  {p}: min={min(vals)} max={max(vals)} "
          f"n_div={24-len(vals)}")