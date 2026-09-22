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


def val(name):
    return M[name]


def num(name):
    s = M[name].strip("$")
    m = re.match(r'([\d.]+)\\times10\^\{(-?\d+)\}$', s)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    return float(s.replace("\\%", ""))


NS = ["Nten", "Ntwentyfive", "Nfifty", "Nhundred"]

print("== disc single-step exact cells ==")
for m in ("Mlp", "Lstm", "Gnn"):
    print(" ", m, [val(f"resSs{m}Disc{n}") for n in NS])

print("\n== OOD ss extremes ==")
extremes = []
for m in ("Mlp", "Lstm", "Gnn"):
    for p in ("Fss", "Ip", "Jg", "Sse", "Seo", "Spm"):
        for n in NS:
            extremes.append((num(f"resSs{m}{p}{n}"), m, p, n))
extremes.sort()
print("  min:", extremes[0])
print("  max:", extremes[-1])
print("  top/bottom 4:", extremes[:4], "...", extremes[-4:])

print("\n== fss ss range (abstract 0.8-3.3%) ==")
fss = sorted((num(f"resSs{m}Fss{n}"), m, n) for m in ("Mlp", "Lstm", "Gnn")
             for n in NS)
print("  min:", fss[0], " max:", fss[-1])

print("\n== GNN never crosses persistence within K=128 (tab:horizon) ==")
for m in ("Mlp", "Lstm", "Gnn"):
    for tag in ("", "St"):
        ks = [val(f"resPh{m}{tag}{n}") for n in NS]
        rs = [num(f"resPhRatio{m}{tag}{n}") for n in NS]
        print(f"  {m}{tag}: k*={ks} ratio_K128={rs}")

print("\n== compounding factor range (abstract x4.7-x6.3) ==")
for m in ("Mlp", "Lstm", "Gnn"):
    for n in NS:
        print(f"  {m} {n}: {val(f'resCompFactor{m}{n}')}", end="  ")
    print()