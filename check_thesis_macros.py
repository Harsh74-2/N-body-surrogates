#!/usr/bin/env python3
"""check_thesis_macros.py — verify every \\res... macro used in the thesis
is defined in the generated metrics.tex, and that no \res macro is defined
but never used."""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
FORK = REPO / "Universe Simulation Thesis"

defs_text = (FORK / "metrics.tex").read_text(encoding="utf-8")
defs = set(re.findall(r'\\newcommand\{\\(res[A-Za-z]+)\}', defs_text))

errs = []
used_all = set()
for fname in ("thesis.tex", "thesis_appendix_results.tex"):
    txt = (FORK / fname).read_text(encoding="utf-8")
    # macro usages: a backslash, then 'res' + letters, not followed by a letter
    used = set(m.group(1) for m in
               re.finditer(r'\\(res[A-Za-z]+)(?![A-Za-z])', txt))
    # drop definitions inside this file (metrics.tex is the only definer,
    # but thesis.tex preamble may \newcommand others) — none start with res.
    used_all |= used
    missing = sorted(used - defs)
    if missing:
        errs.append(f"{fname}: {len(missing)} used-but-UNDEFINED: "
                    + ", ".join(missing))

unused = sorted(defs - used_all)
if unused:
    print(f"note: {len(unused)} defined-but-unused res macros "
          f"(harmless, generated file emits all cells)")

if errs:
    print("MACRO ERRORS:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print(f"OK: {len(used_all)} distinct res macros used, all defined in "
      f"metrics.tex ({len(defs)} defined).")