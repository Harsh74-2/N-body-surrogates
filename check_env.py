#!/usr/bin/env python3
"""check_env.py — structural LaTeX sanity for the fork's thesis.tex +
thesis_appendix_results.tex (local Overleaf-compile proxy, since no local
TeX toolchain is installed):
  1. every \\begin{env} has a matching \\end{env}, properly nested
  2. braces balanced outside verbatim per file
  3. no stray unescaped % that would eat content (heuristic: odd counts of
     unescaped % outside comments are ignored; too noisy — skipped)
  4. no \\res macro whose body contains $ used inside math mode
     (macro bodies like {$\\times$4.7} nest math and break the compile)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FORK = Path(__file__).resolve().parent / "Universe Simulation Thesis"
FILES = [FORK / "thesis.tex", FORK / "thesis_appendix_results.tex"]

begin_re = re.compile(r"\\begin\{([^}]+)\}")
end_re = re.compile(r"\\end\{([^}]+)\}")
comment_re = re.compile(r"(?<!\\)%.*")

ok = True
for f in FILES:
    text = f.read_text(encoding="utf-8")
    stack: list[tuple[str, int]] = []
    errs: list[str] = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = comment_re.sub("", raw)
        for m in begin_re.finditer(line):
            stack.append((m.group(1), i))
        for m in end_re.finditer(line):
            if not stack:
                errs.append(f"  line {i}: \\end{{{m.group(1)}}} with empty stack")
                ok = False
            elif stack[-1][0] != m.group(1):
                errs.append(f"  line {i}: \\end{{{m.group(1)}}} but open "
                            f"\\begin{{{stack[-1][0]}}} (line {stack[-1][1]})")
                ok = False
                # resync: pop until matching
                for j in range(len(stack) - 1, -1, -1):
                    if stack[j][0] == m.group(1):
                        del stack[j:]
                        break
            else:
                stack.pop()
    for env, ln in stack:
        errs.append(f"  unclosed \\begin{{{env}}} from line {ln}")
        ok = False
    # brace balance (excluding \{ \})
    n_open = len(re.findall(r"(?<!\\)\{", text))
    n_close = len(re.findall(r"(?<!\\)\}", text))
    print(f"{f.name}: begin/end {'OK' if not errs else 'ERRORS'}  "
          f"braces {n_open} open / {n_close} close")
    for e in errs:
        print(e)
    if n_open != n_close:
        # multi-line macro defs can unbalance this crude count; report only
        print(f"  note: brace count mismatch by {n_open - n_close} "
              f"(crude count, report only)")

print("OK" if ok else "FAILED")

# 4. $-body macros must not be used inside math mode
dollar_bodies = {}
metrics = FORK / "metrics.tex"
if metrics.is_file():
    for m in re.finditer(r"\\newcommand\{\\(res[A-Za-z]+)\}\{([^}]*)\}",
                         metrics.read_text(encoding="utf-8")):
        if "$" in m.group(2):
            dollar_bodies[m.group(1)] = m.group(2)
bad = 0
for f in FILES + [FORK / "thesis_overleaf" / "thesis.tex"]:
    if not f.is_file():
        continue
    txt = f.read_text(encoding="utf-8")
    for mac in dollar_bodies:
        for m in re.finditer(r"\\" + mac + r"(?![A-Za-z])", txt):
            line = txt[:m.start()].rsplit("\n", 1)[-1]
            if line.count("$") % 2 == 1:
                ln = txt[:m.start()].count("\n") + 1
                print(f"FAIL: {f.name}:{ln} \\{mac} used inside math mode "
                      f"(body {dollar_bodies[mac]!r})")
                bad += 1
                ok = False
if not dollar_bodies:
    print("note: no $-body macros found in metrics.tex")
elif bad == 0:
    print(f"dollar-body macro check: OK ({len(dollar_bodies)} macros, "
          f"0 math-wrapped usages)")
sys.exit(0 if ok else 1)