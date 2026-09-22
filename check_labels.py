#!/usr/bin/env python3
"""check_labels.py — verify every \\cref/\\Cref/\\ref target in the fork's
thesis.tex + thesis_appendix_results.tex resolves to a \\label in the same
two files (comma-lists expanded, duplicates collapsed)."""
from __future__ import annotations

import re
from pathlib import Path

FORK = Path(__file__).resolve().parent / "Universe Simulation Thesis"
FILES = [FORK / "thesis.tex", FORK / "thesis_appendix_results.tex"]

label_re = re.compile(r'\\label\{([^}]+)\}')
ref_re = re.compile(r'\\(?:c|C|v|V|n)?ref\{([^}]+)\}')

labels: set[str] = set()
refs: dict[str, list[tuple[str, int]]] = {}

for f in FILES:
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        for m in label_re.finditer(line):
            labels.add(m.group(1).strip())
        for m in ref_re.finditer(line):
            for tgt in m.group(1).split(","):
                tgt = tgt.strip()
                if tgt:
                    refs.setdefault(tgt, []).append(
                        (f.name, i))

missing = {t: locs for t, locs in refs.items() if t not in labels}
unused = sorted(l for l in labels if l not in refs)

print(f"{len(labels)} labels, {len(refs)} distinct ref targets")
if missing:
    print("UNDEFINED REFERENCES:")
    for t, locs in sorted(missing.items()):
        print(f"  {t}  (used at {locs[0][0]}:{locs[0][1]} +"
              f"{len(locs)-1} more)" if len(locs) > 1 else
              f"  {t}  (used at {locs[0][0]}:{locs[0][1]})")
    raise SystemExit(1)
print("OK: all references defined")
if unused:
    print("defined but never referenced (note only):")
    for u in unused:
        print("  " + u)