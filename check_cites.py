#!/usr/bin/env python3
"""check_cites.py — bibliography audit for the fork:
  1. every cite key used in thesis.tex + thesis_appendix_results.tex
     (\\cite/\\parencite/\\textcite/\\citeauthor/\\citeyear/\\citeyearpar,
     comma-lists expanded) exists in biblio.bib
  2. every bib entry key is unique
  3. report unused bib entries (note only)
"""
from __future__ import annotations

import re
from pathlib import Path

FORK = Path(__file__).resolve().parent / "Universe Simulation Thesis"
FILES = [FORK / "thesis.tex", FORK / "thesis_appendix_results.tex"]
BIB = FORK / "biblio.bib"

cite_re = re.compile(
    r"\\(?:[a-zA-Z]*cite[a-zA-Z]*|footcite|Textcite)\s*(?:\[[^\]]*\])*\{([^}]+)\}")
key_re = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)

used: set[str] = set()
for f in FILES:
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("%"):
            continue
        for m in cite_re.finditer(line):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    used.add(k)

bib_text = BIB.read_text(encoding="utf-8")
keys = key_re.findall(bib_text)
dups = sorted({k for k in keys if keys.count(k) > 1})

missing = sorted(used - set(keys))
unused = sorted(set(keys) - used)

print(f"{len(keys)} bib entries, {len(used)} distinct keys cited")
ok = True
if dups:
    ok = False
    print("DUPLICATE BIB KEYS: " + ", ".join(dups))
if missing:
    ok = False
    print("CITE KEYS MISSING FROM biblio.bib:")
    for k in missing:
        print("  " + k)
if ok:
    print("OK: every citation resolves")
if unused:
    print(f"note: {len(unused)} bib entries never cited (harmless):")
    for u in unused:
        print("  " + u)
raise SystemExit(0 if ok else 1)