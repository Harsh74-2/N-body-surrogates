#!/usr/bin/env python3
"""run_second_model_audit.py — send the thesis audit digest to a local
Ollama model over the HTTP API (no shell pipes) and save the verdicts.

Reads:   audit_prompt.txt + gemini_audit_input.md
Writes:  gemini_audit_output.md
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
MODEL = "qwen3.6"
HOST = "http://localhost:11434"

prompt = (REPO / "audit_prompt.txt").read_text(encoding="utf-8").strip()
digest = (REPO / "gemini_audit_input.md").read_text(encoding="utf-8")
full = prompt + "\n\n" + digest

payload = json.dumps({
    "model": MODEL,
    "prompt": full,
    "stream": True,
    "options": {"temperature": 0.1},
}).encode("utf-8")

req = urllib.request.Request(
    HOST + "/api/generate", data=payload,
    headers={"Content-Type": "application/json"})

chunks: list[str] = []
try:
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            obj = json.loads(raw.decode("utf-8"))
            piece = obj.get("response", "")
            chunks.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
            if obj.get("done"):
                break
except Exception as exc:  # noqa: BLE001
    print(f"\nAPI ERROR: {exc}")
    sys.exit(1)

out = REPO / "gemini_audit_output.md"
out.write_text("".join(chunks), encoding="utf-8")
print(f"\n\nwrote {out} ({out.stat().st_size:,} bytes)")