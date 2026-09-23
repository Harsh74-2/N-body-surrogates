#!/usr/bin/env python3
"""check_zip.py — verify thesis_overleaf.zip:
  1. required root entries present (thesis.tex, main.tex, metrics.tex,
     thesis_appendix_results.tex, biblio.bib, pics/logo.pdf)
  2. every includegraphics target in the zipped thesis.tex +
     thesis_appendix_results.tex resolves to an entry in the zip
  3. metrics.tex in the zip is byte-identical to the fork's
  4. thesis.tex == main.tex, and thesis.tex/appendix are byte-identical
     to the fork's thesis_overleaf copies
  5. no real_case_validation entries outside report_N25
  6. every \res macro used by the zipped tex is defined in zipped metrics.tex
  7. zipped biblio.bib identical to fork's
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
FORK = REPO / "Universe Simulation Thesis"
ZIP = REPO / "thesis_overleaf.zip"

zf = zipfile.ZipFile(ZIP)
names = set(zf.namelist())
ok = True


def fail(msg: str) -> None:
    global ok
    ok = False
    print("FAIL: " + msg)


# 1. required entries
required = ["thesis.tex", "main.tex", "metrics.tex",
            "thesis_appendix_results.tex", "biblio.bib", "pics/logo.pdf"]
for r in required:
    if r not in names:
        fail(f"missing required entry {r}")

# 4. byte-identity vs fork
pairs = [("thesis.tex", FORK / "thesis_overleaf" / "thesis.tex"),
         ("main.tex", FORK / "thesis_overleaf" / "thesis.tex"),
         ("metrics.tex", FORK / "metrics.tex"),
         ("thesis_appendix_results.tex",
          FORK / "thesis_appendix_results.tex"),
         ("biblio.bib", FORK / "biblio.bib")]
for arc, src in pairs:
    if not src.is_file():
        fail(f"byte-identity source missing on disk: {src}")
    if arc not in names:
        continue
    if zf.read(arc) != src.read_bytes():
        fail(f"{arc} differs from {src.name}")

# 2. graphics resolution inside the zip
pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
figs: set[str] = set()
for arc in ("thesis.tex", "thesis_appendix_results.tex"):
    if arc not in names:
        continue
    for line in zf.read(arc).decode("utf-8").splitlines():
        if line.lstrip().startswith("%"):
            continue
        for m in pat.finditer(line):
            figs.add(m.group(1).strip())
missing = []
for p in sorted(figs):
    cands = [p, f"figures/{p}", f"pics/{p}", f"real_case_validation/{p}"]
    if not any(c in names for c in cands):
        missing.append(p)
if missing:
    fail(f"{len(missing)} graphics unresolved in zip: " + ", ".join(missing[:10]))

# 5. rcv scope
rcv = [n for n in names if n.startswith("real_case_validation/")]
bad_rcv = [n for n in rcv if not n.startswith("real_case_validation/report_N25/")]
if bad_rcv:
    fail(f"{len(bad_rcv)} rcv entries outside report_N25, e.g. {bad_rcv[0]}")

# 6. res macros
defs = set(re.findall(r"\\newcommand\{\\(res[A-Za-z]+)\}",
                      zf.read("metrics.tex").decode("utf-8"))) \
    if "metrics.tex" in names else set()
used: set[str] = set()
for arc in ("thesis.tex", "thesis_appendix_results.tex"):
    if arc in names:
        used |= set(re.findall(r"\\(res[A-Za-z]+)(?![A-Za-z])",
                               zf.read(arc).decode("utf-8")))
undef = sorted(used - defs)
if undef:
    fail(f"{len(undef)} res macros used but undefined: {undef[:8]}")

n_png = sum(1 for n in names if n.lower().endswith(".png"))
n_rc_png = sum(1 for n in rcv if n.lower().endswith(".png"))

print(f"entries: {len(names)}  pngs: {n_png}  rcv pngs (N25 grid): {n_rc_png}")
print(f"graphics referenced: {len(figs)}  macros used: {len(used)} "
      f"defined: {len(defs)}")
size = ZIP.stat().st_size
print(f"zip size: {size:,} bytes ({size/1e6:.1f} MB) "
      f"{'OK' if size < 50_000_000 else 'OVER 50MB CAP'}")
print("OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)