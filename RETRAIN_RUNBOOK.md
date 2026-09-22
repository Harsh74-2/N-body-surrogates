# Retrain runbook — one-shot, gated (2026-09-21)

Target VM: 20 cores / 64 GB RAM / RTX 6000 Ada 48 GB, fresh clone
(history rewritten — clone from GitHub, do NOT rsync an old working tree:
stale `training_runs/` checkpoints make both training scripts SKIP cells
silently, and old checkpoints are incompatible with the current code).

```bash
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
git clone <repo> && cd Universe-Simulation
pip install -r requirements.txt
```

## Stage 0 — no data precheck on a fresh clone (expected)

`raw_data/` and `ml_ready_data/` are NOT git-tracked (~2.8 GB total), so a
fresh clone has empty data directories and

```bash
python verify_retrain_gates.py --precheck
```

reports 0 PASS / 12 FAIL (`raw sims missing`). That is EXPECTED here, not a
data problem — the sweep regenerates both trees on the VM. The precheck
becomes meaningful per-cell once the data exists:

```bash
python verify_retrain_gates.py --precheck --N 10 25   # after the probe sweep
python verify_retrain_gates.py --precheck             # after the full grid
```

STOP if any cell that HAS been generated FAILs — do not train on an
unverified dataset.

## Stage 1 — probe cells (~1 h GPU, includes data generation)

The two cheapest cells first; they carry the identity-collapse gate before
any expensive GPU time is spent. The sweep's steps 1–2 also CREATE
`raw_data/N{10,25}/{mlp,lstm,gnn}/` + the npz exports (skipped per-cell if
already present):

```bash
python scaling_sweep.py --N 10 25          # single-step MLP/LSTM/GNN + eval
```

After this completes, run the staged precheck:

```bash
python verify_retrain_gates.py --precheck --N 10 25
```

6 cells PASS = the freshly generated sims + npz/sidecars are consistent
(the npz CRC-full read also guards against a truncated export like the
repaired N100/mlp file).

MLP N=10 and N=25 complete first (~minutes each). Sanity-read the logs:

- epoch 1 must show `roll=0.0000e+00` (warmup_frac=0.5: pure MSE during the
  first half of training, ramp starts at epoch 50/25/… for 100/80/50-epoch
  models) and the ramp must appear afterwards;
- final line must say `selected on val_mse, post-ramp epochs >= 75/60/37`
  (MLP 100 / LSTM 80 / GNN 50): best-ckpt selection is RESTRICTED to
  post-ramp epochs, so the saved stable checkpoint always experienced the
  fully-active rollout loss (adversarial-crosscheck fix, 2026-09-21).

## Stage 2 — probe gate

```bash
python verify_retrain_gates.py --N 10 25 --allow-missing
```

- The HARD gate is `mse < mse_identity` (beat the persistence floor):
  EV ≈ 0 or negative ⇒ identity collapse ⇒ STOP, do not launch the grid.
  Explained variance **> 0.5** is expected for MLP/LSTM; for the GNN a low
  EV with mse still below identity is a GENERALIZATION gap, not a collapse
  (2026-09-21 probe: GNN N=10 EV 0.05, mse 2.71e-06 vs identity 2.86e-06,
  train/val gap ~500x — real result, reported not blocking). The gate
  prints this case as a warning.
- `stability.json` records are MISSING at this point; `--allow-missing`
  accounts for that. MISSING cells for `metrics_*.json` are NOT allowed
  and still fail — a probe cell that produced no eval artifact is a
  hard error.
- GO/NO-GO: only if all present cells PASS, continue. (A probe FAIL costs
  ~1 GPU-hour; a grid-wide collapse costs 24 GPU-hours.)

## Stage 3 — full 24-cell grid (~23–24 GPU-h sequential; ~1.5 days)

DELETE STALE COLLAPSED PROBE OUTPUTS FIRST (2026-09-21 rework): the probe
cells trained with the OLD rollout-ENERGY code are identity-collapsed, and
the sweep's resumability would SKIP them silently. Data directories are
fine — the rollout-MSE targets are derived from the npz at load time, so
no data regeneration is needed:

```bash
rm -rf training_runs/N10 training_runs/N25 results/N10 results/N25
```

Grid launch (unchanged from the original plan):

```bash
# finish the single-step cells for all N (N=50, 100 + remaining models);
# this also generates raw_data + npz for N=50/100 (skips what exists)
python scaling_sweep.py

# stable variants — GNN N=100 stable trains at b=96 (only that cell),
# expandable_segments is exported by the script itself
bash train_stable_variants.sh
```

After the full sweep's data generation, the full precheck must PASS before
trusting the remaining cells:

```bash
python verify_retrain_gates.py --precheck    # 12/12 PASS expected now
```

- Resumable: completed cells are skipped via their `model_best.pt`.
- One failing cell is recorded and the queue continues; re-run the script
  to retry failed cells.
- Stable cells now train the rollout-MSE term (w_energy 0, w_rollout 0.1)
  and select on val_total post-ramp; single-step cells keep val_mse — the
  final line reads `selected on val_total` for stable, `selected on
  val_mse` for single-step (rollout-MSE rework, 2026-09-21).
- Expected wall-clock: `gnn_stable` N=100 is the long pole (~13–15 h at
  b=96); run cheap cells concurrently only while no big GNN cell is
  training.

## Stage 4 — rollout stability benchmark (both variants, every N)

One invocation per N with all six checkpoints — the script writes ALL of
them into a single `results/N{n}/stability.json` (list of per-model
records), which is what the gate reads. Each record carries the
robust-statistics fields the final gate consumes (2026-09-21): per-start
model/persistence ratios → median + max, and the spatial-collapse guard
`collapse_guard.spatial_var_ratio_final`:

```bash
for N in 10 25 50 100; do
  python stability_benchmark.py \
    --ckpt training_runs/N${N}/mlp/model_best.pt:mlp \
    --ckpt training_runs/N${N}/lstm/model_best.pt:lstm \
    --ckpt training_runs/N${N}/gnn/model_best.pt:gnn \
    --ckpt training_runs/N${N}/mlp_stable/model_best.pt:mlp \
    --ckpt training_runs/N${N}/lstm_stable/model_best.pt:lstm \
    --ckpt training_runs/N${N}/gnn_stable/model_best.pt:gnn \
    --N ${N} --K 128 --rollout-batches 16 \
    --json results/N${N}/stability.json \
    || echo "FAILED N=${N}"
done
```

## Stage 5 — final gate (BLOCKING)

```bash
python verify_retrain_gates.py
```

All 24 cells must PASS:
- explained variance > 0.5 (or the mse-beats-identity generalization-gap
  warning path),
- rollout model/persistence **MEDIAN** ratio < 1.0 and **MAX per-start**
  ratio < 100 (robust stats, 2026-09-21: a raw mean is infinitely
  sensitive to one exploding rollout among healthy ones),
- spatial_var_ratio_final >= 0.5 — **G6, centre-of-mass collapse guard**
  (2026-09-21): a clamped-to-CoM model scores the dataset spatial
  variance as its MSE and at long K the persistence floor degrades past
  that variance, so the ratio alone can pass a physically dead model;
- param counts = canonical (MLP 210,182 / LSTM / GNN 168,199 — printed
  by the script), all metrics finite.
Nonzero exit = at least one cell failed; investigate BEFORE spending
time on downstream regeneration. NOTE: stability.json files produced by
a pre-2026-09-21 benchmark (no median/max/collapse_guard) FAIL by design
— re-run the Stage 4 benchmark with the updated script.

### Stage 5 waiver — LSTM N=100 G2 failures (documented, 2026-09-22)

The 2026-09-21/22 retrain produced 22 PASS / 2 FAIL / 0 MISSING: both
LSTM N=100 cells fail G2 (median model/persistence ratio 23.7 / 10.1,
max 373 / 198). Diagnosis (per-step ratio curves): the cells beat
persistence by 3–5× up to k=32 (ratio@32 0.37 / 0.22), cross over at
k=39 / 45 of 128, and have no blow-up (`divergence_step = None`); G6
passed. This is genuine late-horizon error compounding in the more
chaotic N=100 regime — the persistence anchor is the strongest baseline
for long chaotic rollouts, and the cross-N trend is monotone (N=25
crossover k=121/127 → N=50 never crosses → N=100 k=39/45). Stability
training halves the compounding (median 23.7→10.1, crossover 39→45).
Gemini-reviewed verdict (2026-09-22): keep G2 exactly as-is, record the
2 FAILs as an architectural-limit finding, waive with documented
rationale — do NOT weaken the gate post hoc. The waiver is explicit and
recorded into the verdict JSON (thresholds untouched):

```bash
python verify_retrain_gates.py \
  --waive N100/lstm_single_step --waive N100/lstm_stable \
  --waive-reason "LSTM N=100 late-horizon compounding: single-step best-in-class (5.3e-08, EV 0.98), beats persistence to k=32 (ratio 0.37/0.22), crossover k=39/45 of 128, no blowup, G6 pass; stable halves compounding (median 23.7->10.1) — architectural capacity limit at the most chaotic N, documented thesis finding (predictive_horizon.md)" \
  --json-out results/retrain_gates.json
```

Exit 0 with `22 PASS / 0 FAIL / 0 MISSING / 2 WAIVED` + `verdict: PASS`
unblocks Stage 6. The thesis table is generated from the SAME records:

```bash
python predictive_horizon_report.py
```

(Also see the K=32 falsification check in Stage 6 step 0 below.)

## Stage 6 — downstream regeneration (only after Stage 5 PASSES)

0. Falsification check on the waived cells (cheap, ~1 min): re-run the
   two waived LSTM N=100 cells at K=32 — the crossover analysis predicts
   median ratio < 1 there (crossover k=39/45 > 32). A median >= 1 at K=32
   would mean non-monotonic error growth before the crossover — i.e. a
   hidden-state/protocol bug, NOT an architectural limit — STOP and
   re-diagnose in that case. Write to a separate file so the K=128 gate
   input stays untouched:

```bash
python stability_benchmark.py \
  --ckpt training_runs/N100/lstm/model_best.pt:lstm \
  --ckpt training_runs/N100/lstm_stable/model_best.pt:lstm \
  --N 100 --K 32 --rollout-batches 16 \
  --json results/N100/stability_k32.json
```

1. OOD real-case grid (28 preset×variant cells) + single-step dumps (28).
   Before `make_site_audits.py`, `results/all_eval.json` must exist — it
   is NOT produced by any training script; assemble it from the per-cell
   eval outputs (also updates the make_site_audits ordering anchors —
   cross-check the printed gnn N=25/N=100 mse values against the
   `anchors` constants in make_site_audits.py):

```bash
python make_all_eval.py
```
2. `regen_top_level_plots.py`, `run_animations.py` /
   `render_animations_parallel.py` (168 mp4s).
3. Stability plots + aggregate; cross-N audit mds; site audits.
4. Thesis numbers reconciliation (param counts, tables, identity-floor
   note) + thesis_bundle/zip rebuild — remember: `cp` the private fork's
   `thesis_overleaf/` into the public root before EVERY zip rebuild.