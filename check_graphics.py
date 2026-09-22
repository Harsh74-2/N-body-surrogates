#!/usr/bin/env python3
"""check_graphics.py — verify every includegraphics target in the fork's
thesis.tex + thesis_appendix_results.tex exists on disk (figures/, pics/,
real_case_validation/, or the fork root)."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
FORK = REPO / "Universe Simulation Thesis"
FILES = [FORK / "thesis.tex", FORK / "thesis_appendix_results.tex"]

pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

figs: set[str] = set()
for f in FILES:
    lines = f.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.lstrip().startswith("%"):
            continue  # comment block
        for m in pat.finditer(line):
            figs.add(m.group(1).strip())

missing = sorted(
    p for p in figs
    if not (FORK / "figures" / p).exists()
    and not (FORK / "pics" / p).exists()
    and not (FORK / p).exists()
)

print(f"{len(figs)} unique graphics referenced")
if missing:
    print("MISSING:")
    for m in missing:
        print("  " + m)
    raise SystemExit(1)
print("OK: all graphics resolve on disk")