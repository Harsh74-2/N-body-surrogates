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
    return float(s.replace("\\%", ""))


NS = ["Nten", "Ntwentyfive", "Nfifty", "Nhundred"]
win = 0
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        a, b = num(f"res{m}Roll{n}"), num(f"res{m}StRoll{n}")
        w = b < a
        win += w
        print(f"  {m} {n}: single={a:.3g} stable={b:.3g} stable-better={w}")
print(f"K=50 rollout drift stable-better: {win}/12")