#!/usr/bin/env python3
"""make_metrics_tex.py — auto-generate metrics.tex macro file from the fresh
2026-09-22 post-retrain artifacts.

Every thesis number is emitted as a namespaced \\newcommand macro
(prefix \\res...) with a pre-formatted LaTeX value, so tables and prose
never hand-transcribe a number. Sources:

  results/all_eval.json                       in-dist single-step/rollout/identity/EV/latency (24 cells)
  real_case_validation/report_N*/preset_*/summary.json
                                              OOD rollout, calibrated + leapfrog lanes
  real_case_validation/report_N*/single_step/preset_*/ss_summary.json
                                              single-step per-preset lane
  results/stability_summary.json              K=128 rollout slopes
  results/predictive_horizon.json             predictive horizon k* vs persistence
  results/latency_bench.json                  solver/surrogate latency (40 repeats, CPU)

Self-verification: the emitted OOD rollout grid, OOD means, single-step
grid, no-compounding means and compounding factors are compared
cell-by-cell against the canonical audit tables in
results/cross_N_audit.md and results/cross_N_audit_single_step.md;
any mismatch is reported and exits non-zero.

Writes:  Universe Simulation Thesis/metrics.tex
         results/metrics_macros.json   (macro -> raw value -> source, audit trail)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
FORK = REPO / "Universe Simulation Thesis"
RCV = REPO / "real_case_validation"

PRESETS = {
    "Disc": "disc_imf_in_distribution_baseline",
    "Fss": "full_solar_system",
    "Ip": "inner_planets",
    "Jg": "jupiter_galileans",
    "Sse": "solar_system_extended",
    "Seo": "sun_earth_only",
    "Spm": "sun_planets_moon",
}
PRESET_ORDER = ["Disc", "Fss", "Ip", "Jg", "Sse", "Seo", "Spm"]
OOD_PRESETS = ["Fss", "Ip", "Jg", "Sse", "Seo", "Spm"]
NS = {10: "Nten", 25: "Ntwentyfive", 50: "Nfifty", 100: "Nhundred",
      200: "Ntwohundred"}
MODELS = {"mlp": "Mlp", "lstm": "Lstm", "gnn": "Gnn"}
VARIANTS = {"single_step": "", "stable": "St"}


# ---------- formatting ----------
def sci3(v: float) -> str:
    """3 significant figures, math scientific notation."""
    if v == 0:
        return "$0$"
    exp = 0
    x = abs(v)
    while x >= 10:
        x /= 10
        exp += 1
    while x < 1:
        x *= 10
        exp -= 1
    mant = f"{x:.2f}"
    if mant == "10.00":
        mant, exp = "1.00", exp + 1
    return f"${mant}\\times10^{{{exp}}}$"


def pct1(v: float) -> str:
    return f"{v:.1f}\\%"


def dec(v: float, nd: int) -> str:
    return f"{v:.{nd}f}"


def fmt_ms(v: float) -> str:
    if v < 10:
        return f"{v:.3f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def ph_fmt(c: dict) -> str:
    if c["predictive_horizon"] is not None:
        return str(c["predictive_horizon"])
    if c["never_crosses_within_K"]:
        return f">{c['K']}"
    return "??"


# ---------- audit-md parsing for self-verification ----------
def parse_md_table(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def unescape(cell: str) -> str:
    return (cell.replace("&gt;", ">").replace("&lt;", "<")
                .replace("&mdash;", "--").replace("&times;", "x"))


def main() -> int:
    macros: dict[str, str] = {}
    trail: dict[str, dict] = {}

    def put(name: str, value: str, raw, source: str) -> None:
        if name in macros:
            raise SystemExit(f"duplicate macro {name}")
        macros[name] = value
        trail[name] = {"raw": raw, "source": source, "tex": value}

    # ---- 1. in-distribution cells (all_eval.json) ----
    all_eval = json.loads((REPO / "results" / "all_eval.json")
                          .read_text(encoding="utf-8"))
    params = {}
    for c in all_eval:
        n, mt = c["cell"].split("/")
        m, v = mt, c["variant"]
        nt = NS[int(n[1:])]
        mtok, vtok = MODELS[m], VARIANTS[v]
        src = "results/all_eval.json"
        put(f"res{mtok}{vtok}Mse{nt}", sci3(c["mse"]), c["mse"], src)
        put(f"res{mtok}{vtok}Ev{nt}", dec(c["mse_explained_var"], 3),
            c["mse_explained_var"], src)
        put(f"res{mtok}{vtok}Ident{nt}", sci3(c["mse_identity"]),
            c["mse_identity"], src)
        put(f"res{mtok}{vtok}Energy{nt}", sci3(c["energy"]), c["energy"], src)
        put(f"res{mtok}{vtok}Lat{nt}", dec(c["latency_ms"], 3),
            c["latency_ms"], src)
        put(f"res{mtok}{vtok}Roll{nt}", sci3(c["rollout"]), c["rollout"], src)
        params[mtok] = c["n_params"]
    for mtok, p in params.items():
        put(f"resParams{mtok}", f"{p:,}", p, "results/all_eval.json")

    # ---- 2. OOD rollout (calibrated + leapfrog lanes) ----
    # variants in the rollout lane: all six
    RV = {"mlp": "Mlp", "mlp_stable": "MlpSt", "lstm": "Lstm",
          "lstm_stable": "LstmSt", "gnn": "Gnn", "gnn_stable": "GnnSt"}
    roll: dict[tuple[str, str, str], dict] = {}
    RVJSON = {"MLP": "mlp", "MLP_stable": "mlp_stable", "LSTM": "lstm",
              "LSTM_stable": "lstm_stable", "GNN": "gnn",
              "GNN_stable": "gnn_stable"}
    for n in (10, 25, 50, 100):
        for pkey, pd in PRESETS.items():
            f = RCV / f"report_N{n}" / f"preset_{pd}" / "summary.json"
            j = json.loads(f.read_text(encoding="utf-8"))
            nt = NS[n]
            for mv, mvj in j["per_model"].items():
                mv = RVJSON[mv]
                mtok = RV[mv]
                src = str(f.relative_to(REPO)).replace("\\", "/")
                base = f"resRoll{mtok}{pkey}{nt}"
                if pkey == "Disc":
                    # in-distribution baseline: no calibrated lane exists;
                    # the audit grid scores it against the leapfrog reference
                    raw_mean = mvj["mean_error_over_L"] * 100
                    cal = {"mean_err_pct": raw_mean,
                           "max_err_pct": mvj["max_error_over_L"] * 100}
                else:
                    cal = mvj["calibrated"]
                    raw_mean = mvj["mean_error_over_L"] * 100
                roll[(mv, pkey, str(n))] = cal["mean_err_pct"]
                if cal["mean_err_pct"] >= 100.0:
                    put(base + "Mean", "div.", cal["mean_err_pct"], src)
                else:
                    put(base + "Mean", pct1(cal["mean_err_pct"]),
                        cal["mean_err_pct"], src)
                if cal["max_err_pct"] >= 100.0:
                    put(base + "Max", "div.", cal["max_err_pct"], src)
                else:
                    put(base + "Max", pct1(cal["max_err_pct"]),
                        cal["max_err_pct"], src)
                put(base + "RawMean", pct1(raw_mean), raw_mean, src)
                put(base + "Frames", str(mvj["frames_before_half_L"]),
                    mvj["frames_before_half_L"], src)
                put(base + "Energy", sci3(mvj["max_energy_drift"]),
                    mvj["max_energy_drift"], src)

    # survivor OOD mean (calibrated < 100, six OOD presets) -- matches
    # results/cross_N_audit.md headline convention
    ood_mean: dict[tuple[str, str], float] = {}
    for mv, mtok in RV.items():
        for n in (10, 25, 50, 100):
            vals = [roll[(mv, p, str(n))] for p in OOD_PRESETS
                    if roll[(mv, p, str(n))] < 100.0]
            nt = NS[n]
            if not vals:
                put(f"resRollOOD{mtok}{nt}", ">100", None,
                    "derived: all 6 OOD presets diverged")
            else:
                m = sum(vals) / len(vals)
                ood_mean[(mv, str(n))] = m
                put(f"resRollOOD{mtok}{nt}", pct1(m), m,
                    f"derived: survivor mean over {[p for p in OOD_PRESETS if roll[(mv, p, str(n))] < 100.0]}")

    # ---- 3. single-step per-preset lane (no stable variants) ----
    ss: dict[tuple[str, str, str], float] = {}
    SSJSON = {"mlp": "MLP", "lstm": "LSTM", "gnn": "GNN"}
    for n in (10, 25, 50, 100):
        for pkey, pd in PRESETS.items():
            f = RCV / f"report_N{n}" / "single_step" / f"preset_{pd}" / "ss_summary.json"
            j = json.loads(f.read_text(encoding="utf-8"))
            src = str(f.relative_to(REPO)).replace("\\", "/")
            for m, mtok in MODELS.items():
                pm = j["per_model"][SSJSON[m]]
                v = pm["mean_err_pct"]
                ss[(m, pkey, str(n))] = v
                put(f"resSs{mtok}{pkey}{NS[n]}", pct1(v), v, src)
                put(f"resSs{mtok}{pkey}{NS[n]}Max", pct1(pm["max_err_pct"]),
                    pm["max_err_pct"], src)
                put(f"resSs{mtok}{pkey}{NS[n]}Energy", sci3(pm["energy_drift"]),
                    pm["energy_drift"], src)

    # no-compounding OOD mean (mean over all 6 OOD presets)
    nocomp: dict[tuple[str, str], float] = {}
    for m, mtok in MODELS.items():
        for n in (10, 25, 50, 100):
            vals = [ss[(m, p, str(n))] for p in OOD_PRESETS]
            mm = sum(vals) / len(vals)
            nocomp[(m, str(n))] = mm
            put(f"resNoComp{mtok}{NS[n]}", pct1(mm), mm,
                "derived: mean of 6 OOD single-step presets")

    # compounding factor (no-comp vs survivor rollout mean)
    for m, mtok in MODELS.items():
        for n in (10, 25, 50, 100):
            nt = NS[n]
            rk = (m, str(n))
            if rk in ood_mean and rk in nocomp:
                fac = ood_mean[rk] / nocomp[rk]
                put(f"resCompFactor{mtok}{nt}", f"$\\times${fac:.1f}", fac,
                    "derived: rollout survivor mean / no-comp mean")
            else:
                put(f"resCompFactor{mtok}{nt}", "--", None,
                    "derived: one lane missing")

    # ---- 4. stability summary ----
    st = json.loads((REPO / "results" / "stability_summary.json")
                    .read_text(encoding="utf-8"))
    SV = {"mlp_single_step": "Mlp", "mlp_stable": "MlpSt",
          "lstm_single_step": "Lstm", "lstm_stable": "LstmSt",
          "gnn_single_step": "Gnn", "gnn_stable": "GnnSt"}
    for n_str, variants in st.items():
        nt = NS[int(n_str)]
        for vk, vj in variants.items():
            tok = SV[vk]
            src = "results/stability_summary.json"
            put(f"resStab{tok}MseSlope{nt}", sci3(vj["mse_slope"]),
                vj["mse_slope"], src)
            put(f"resStab{tok}ESlope{nt}", sci3(vj["energy_drift_slope"]),
                vj["energy_drift_slope"], src)
            put(f"resStab{tok}MoiMean{nt}", dec(vj["model_over_identity_mean"], 2),
                vj["model_over_identity_mean"], src)
            put(f"resStab{tok}MoiMax{nt}", dec(vj["model_over_identity_max"], 2),
                vj["model_over_identity_max"], src)
            put(f"resStab{tok}Spatial{nt}",
                dec(vj["spatial_var_ratio_final"], 2),
                vj["spatial_var_ratio_final"], src)
            ds = vj["divergence_step"]
            put(f"resStab{tok}DivStep{nt}",
                "never" if ds is None else str(ds), ds, src)

    # ---- 5. predictive horizon ----
    ph = json.loads((REPO / "results" / "predictive_horizon.json")
                    .read_text(encoding="utf-8"))
    for c in ph["cells"]:
        m, tok = c["model"], MODELS[c["model"]]
        vtok = "" if c["variant"] == "single_step" else "St"
        nt = NS[c["N"]]
        src = "results/predictive_horizon.json"
        put(f"resPh{tok}{vtok}{nt}", ph_fmt(c), c["predictive_horizon"], src)
        r = c["ratio_k128"]
        put(f"resPhRatio{tok}{vtok}{nt}", dec(r, 2), r, src)

    # ---- 6. latency bench ----
    lb = json.loads((REPO / "results" / "latency_bench.json")
                    .read_text(encoding="utf-8"))
    src = "results/latency_bench.json"
    for n_str, v in lb["solver"].items():
        put(f"resLatBenchSolver{NS[int(n_str)]}", fmt_ms(v), v, src)
    for m, mtok in MODELS.items():
        for n_str, v in lb["surrogate_single_frame"][m].items():
            put(f"resLatBench{mtok}Single{NS[int(n_str)]}", fmt_ms(v), v, src)
        for n_str, v in lb["surrogate_batched_amortised"][m].items():
            put(f"resLatBench{mtok}Batched{NS[int(n_str)]}", fmt_ms(v), v, src)
    for lane, key in (("Single", "crossover_Nstar_single_frame"),
                      ("Batched", "crossover_Nstar_batched")):
        for m, v in lb[key].items():
            put(f"resLatCross{MODELS[m]}{lane}",
                "never" if v is None else str(v), v, src)

    # ---- SELF-VERIFICATION against canonical audit mds ----
    errs: list[str] = []

    # (a) rollout grid + headline OOD mean vs cross_N_audit.md
    grid_rows = parse_md_table(REPO / "results" / "cross_N_audit.md")
    # headline table: | model | N=10 | N=25 | N=50 | N=100 |
    head = {}
    for r in grid_rows:
        if len(r) == 5 and r[0].startswith(("MLP", "LSTM", "GNN")):
            head[r[0]] = r[1:]
    for label, mv in (("MLP", "mlp"), ("MLP (stable)", "mlp_stable"),
                      ("LSTM", "lstm"), ("LSTM (stable)", "lstm_stable"),
                      ("GNN", "gnn"), ("GNN (stable)", "gnn_stable")):
        for i, n in enumerate((10, 25, 50, 100)):
            want = unescape(head[label][i])
            if want == ">100":
                want_macro = macros[f"resRollOOD{RV[mv]}{NS[n]}"]
                if want_macro != ">100":
                    errs.append(f"OOD mean {label} N{n}: md '>100' vs macro '{want_macro}'")
            else:
                got = macros[f"resRollOOD{RV[mv]}{NS[n]}"]
                if got != pct1(float(want.replace("%", "").replace("\\%", ""))):
                    errs.append(f"OOD mean {label} N{n}: md '{want}' vs macro '{got}'")
    # full grid (10 columns: N, variant, 7 presets, OOD mean)
    start = None
    for i, r in enumerate(grid_rows):
        if r and r[0] == "N" and r[1] == "variant":
            start = i
            break
    if start is None:
        errs.append("cross_N_audit.md: per-preset grid table not found")
        start = len(grid_rows)
    n_grid_rows = 0
    for r in grid_rows[start:]:
        if len(r) != 10 or not r[0].strip().rstrip("=").strip().isdigit():
            continue
        n_grid_rows += 1
        n = int(r[0].lstrip("N="))
        label = r[1]
        mv = {"MLP": "mlp", "MLP (stable)": "mlp_stable",
              "LSTM": "lstm", "LSTM (stable)": "lstm_stable",
              "GNN": "gnn", "GNN (stable)": "gnn_stable"}[label]
        for j, pkey in enumerate(PRESET_ORDER):
            want = unescape(r[2 + j])
            # the md prints the disc baseline raw (it is in-distribution,
            # excluded from the survivor mean); OOD presets use div.
            suffix = "RawMean" if pkey == "Disc" else "Mean"
            mac = f"resRoll{RV[mv]}{pkey}{NS[n]}{suffix}"
            if want in ("div.", "--"):
                if pkey == "Disc":
                    errs.append(f"rollout {label} disc N{n}: md '{want}' but the md prints disc raw")
                elif macros[mac] != "div.":
                    errs.append(f"rollout {label} {pkey} N{n}: md div. vs macro '{macros[mac]}'")
            else:
                v = float(want.replace("%", "").replace("\\%", ""))
                if abs(roll[(mv, pkey, str(n))] - v) > 0.05:
                    errs.append(f"rollout {label} {pkey} N{n}: md {v} vs json {roll[(mv, pkey, str(n))]:.2f}")
                if macros[mac] != pct1(v):
                    errs.append(f"rollout {label} {pkey} N{n}: formatting {macros[mac]} vs md {want}")
        # OOD mean column of the grid must agree with the headline table
        want_ood = unescape(r[9])
        head_vals = head.get(label)
        if head_vals is not None:
            hv = unescape(head_vals[(10, 25, 50, 100).index(n)])
            diverged = {">100", "--", "div."}
            if not (hv == want_ood or (hv in diverged and want_ood in diverged)):
                errs.append(f"rollout {label} N{n}: grid OOD mean '{want_ood}' vs headline '{hv}'")
    if n_grid_rows != 24:
        errs.append(f"cross_N_audit.md: expected 24 grid rows, parsed {n_grid_rows}")

    # (b) single-step per-preset grid vs cross_N_audit_single_step.md.
    # Section headers ("### preset (OOD)") are outside the pipe tables, so
    # parse the raw text with a section tracker instead of parse_md_table.
    ss_md = REPO / "results" / "cross_N_audit_single_step.md"
    section_map = {"disc_imf_in_distribution_baseline (in-distribution)": "Disc",
                   "full_solar_system (OOD)": "Fss",
                   "inner_planets (OOD)": "Ip",
                   "jupiter_galileans (OOD)": "Jg",
                   "solar_system_extended (OOD)": "Sse",
                   "sun_earth_only (OOD)": "Seo",
                   "sun_planets_moon (OOD)": "Spm"}
    cur_preset = None
    n_ss_rows = 0
    for line in ss_md.read_text(encoding="utf-8").splitlines():
        ls = line.strip()
        if ls.startswith("### "):
            cur_preset = section_map.get(ls[4:].strip())
            continue
        if cur_preset is None or not ls.startswith("|"):
            continue
        r = [c.strip() for c in ls.strip("|").split("|")]
        if set("".join(r)) <= set("-: "):
            continue
        if len(r) == 5 and r[0] in ("MLP", "LSTM", "GNN"):
            n_ss_rows += 1
            for i, n in enumerate((10, 25, 50, 100)):
                want = unescape(r[1 + i])
                if want in ("--", "div."):
                    continue
                v = float(want.replace("%", "").replace("\\%", ""))
                jv = ss[(r[0].lower(), cur_preset, str(n))]
                if abs(jv - v) > 0.005:
                    errs.append(f"ss {r[0]} {cur_preset} N{n}: md {v} vs json {jv:.4f}")
                mac = f"resSs{MODELS[r[0].lower()]}{cur_preset}{NS[n]}"
                if mac not in macros:
                    errs.append(f"missing macro {mac}")
                elif macros[mac] != pct1(jv):
                    errs.append(f"ss {r[0]} {cur_preset} N{n}: macro '{macros[mac]}' vs pct1(json) {pct1(jv)}")
    if n_ss_rows != 21:
        errs.append(f"single-step md: expected 21 rows (3 models x 7 presets), parsed {n_ss_rows}")

    # (c) compounding table (cols: N | variant | no compounding | full rollout | factor)
    comp = {}
    in_comp = False
    for r in grid_rows:
        if len(r) == 5 and "compounding" in " ".join(r).lower():
            in_comp = True
            continue
        if in_comp and len(r) == 5:
            label, m = r[1], {"MLP": "mlp", "LSTM": "lstm", "GNN": "gnn"}[r[1].split(" ")[0]]
            n = int(r[0].lstrip("N="))
            comp[(label, n)] = (unescape(r[2]), unescape(r[3]), unescape(r[4]))
    for (label, n), (nc, fr, fac) in comp.items():
        m = label.split(" ")[0].lower()
        is_st = "(stable)" in label
        if not is_st and nc != "--":
            mac = f"resNoComp{MODELS[m]}{NS[n]}"
            if mac not in macros:
                errs.append(f"missing macro {mac}")
            elif macros[mac] != pct1(float(nc.replace("%", "").replace("\\%", ""))):
                errs.append(f"nocomp {label} N{n}: md '{nc}' vs macro '{macros[mac]}'")
        if not is_st and fac != "--":
            mac = f"resCompFactor{MODELS[m]}{NS[n]}"
            want = fac.replace("x", "").replace("×", "").strip()
            got = macros[mac].replace("$\\times$", "").replace("$", "").strip()
            if got != want:
                errs.append(f"factor {label} N{n}: md '{fac}' vs macro '{macros[mac]}'")

    if errs:
        print("SELF-VERIFICATION MISMATCHES:")
        for e in errs:
            print("  -", e)
        return 1
    print(f"SELF-VERIFICATION OK: OOD means ({len(head)} headline rows), rollout grid")
    print(f"  ({n_grid_rows} rows), single-step grid ({n_ss_rows} rows),")
    print(f"  no-compounding means and compounding factors ({len(comp)} rows) all")
    print(f"  match the canonical mds.")
    print(f"  {len(macros)} macros emitted.")

    # ---- write metrics.tex ----
    lines = [
        "% =====================================================================",
        "% metrics.tex — AUTO-GENERATED by make_metrics_tex.py (2026-09-22 run).",
        "% DO NOT EDIT by hand: every macro is read directly from the fresh",
        "% post-retrain artifacts; regenerate with `python make_metrics_tex.py`.",
        "% Sources: results/all_eval.json, results/stability_summary.json,",
        "% results/predictive_horizon.json, results/latency_bench.json,",
        "% real_case_validation/report_N*/{preset_*/summary.json,",
        "% single_step/preset_*/ss_summary.json}.",
        "% Naming: \\res<Model><St?><Metric><Ntoken>, presets Disc/Fss/Ip/Jg/Sse/Seo/Spm,",
        "% N tokens Nten/Ntwentyfive/Nfifty/Nhundred/Ntwohundred.",
        "% =====================================================================",
        "",
    ]
    for name in sorted(macros):
        v = macros[name]
        if "\\" in v or "{" in v:
            body = "{" + v + "}"
        else:
            body = "{" + v + "}"
        lines.append(f"\\newcommand{{\\{name}}}{body}")
    out = FORK / "metrics.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[tex] -> {out}")
    (REPO / "results" / "metrics_macros.json").write_text(
        json.dumps(trail, indent=1), encoding="utf-8")
    print("[json] -> results/metrics_macros.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())