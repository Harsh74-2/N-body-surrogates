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

## Stage 0 — preflight data gate (~2 min, CPU)

```bash
python verify_retrain_gates.py --precheck
```

12 cells PASS = raw sims + npz/sidecars consistent. If the sweep re-exports
the npz on the VM, re-run the precheck AFTER the export step and BEFORE any
training. STOP if anything FAILs — do not train on an unverified dataset.

## Stage 1 — probe cells (~1 h GPU)

The two cheapest cells first; they carry the identity-collapse gate before
any expensive GPU time is spent:

```bash
python scaling_sweep.py --N 10 25          # single-step MLP/LSTM/GNN + eval
```

MLP N=10 and N=25 complete first (~minutes each). Sanity-read the logs:

- epoch 1 must show `roll=0.0000e+00` (warmup_frac=0.5: pure MSE during the
  first half of training, ramp starts at epoch 50/25/… for 100/80/50-epoch
  models) and the ramp must appear afterwards;
- final line must say `selected on val_mse`.

## Stage 2 — probe gate

```bash
python verify_retrain_gates.py --N 10 25 --allow-missing
```

- Explained variance (`expl.var` in the eval table) must be **> 0.5** for
  every probe cell (default gate). EV ≈ 0 or negative ⇒ identity collapse
  ⇒ STOP, do not launch the grid — report the value first.
- `stability.json` records are MISSING at this point; `--allow-missing`
  accounts for that. MISSING cells for `metrics_*.json` are NOT allowed
  and still fail — a probe cell that produced no eval artifact is a
  hard error.
- GO/NO-GO: only if all present cells PASS, continue. (A probe FAIL costs
  ~1 GPU-hour; a grid-wide collapse costs 24 GPU-hours.)

## Stage 3 — full 24-cell grid (~23–24 GPU-h sequential; ~1.5 days)

```bash
# finish the single-step cells for all N (N=50, 100 + remaining models)
python scaling_sweep.py

# stable variants — GNN N=100 stable trains at b=64 (only that cell),
# expandable_segments is exported by the script itself
bash train_stable_variants.sh
```

- Resumable: completed cells are skipped via their `model_best.pt`.
- One failing cell is recorded and the queue continues; re-run the script
  to retry failed cells.
- Expected wall-clock: `gnn_stable` N=100 is the long pole (~15 h at b=64);
  run cheap cells concurrently only while no big GNN cell is training.

## Stage 4 — rollout stability benchmark (both variants, every N)

One invocation per N with all six checkpoints — the script writes ALL of
them into a single `results/N{n}/stability.json` (list of per-model
records), which is what the gate reads:

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

All 24 cells must PASS: explained variance > 0.5, rollout
model/persistence ratio < 1.0, param counts = canonical
(MLP 210,182 / LSTM / GNN 168,199 — printed by the script), all metrics
finite. Nonzero exit = at least one cell failed; investigate BEFORE
spending time on downstream regeneration.

## Stage 6 — downstream regeneration (only after Stage 5 PASSES)

1. OOD real-case grid (28 preset×variant cells) + single-step dumps (28).
2. `regen_top_level_plots.py`, `run_animations.py` /
   `render_animations_parallel.py` (168 mp4s).
3. Stability plots + aggregate; cross-N audit mds; site audits.
4. Thesis numbers reconciliation (param counts, tables, identity-floor
   note) + thesis_bundle/zip rebuild — remember: `cp` the private fork's
   `thesis_overleaf/` into the public root before EVERY zip rebuild.