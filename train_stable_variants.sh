#!/usr/bin/env bash
# train_stable_variants.sh
# ========================
# Retrain the three surrogates (MLP / LSTM / GNN) with the K-step rollout
# MSE loss ENABLED (w_rollout = 0.1, w_energy = 0): the "stability-trained"
# variant contrasted against the single-step-trained checkpoints in the
# stability benchmark.
#
# REDESIGN (2026-09-21, cross-verified): the previous rollout-ENERGY term
# has the identity function as its global optimum — a frozen state
# conserves energy exactly, so |E_k − E0| ≡ 0 — and once the ramp
# completed, every variant was dragged back to the persistence floor
# (the September-2026 identity collapse; the probe caught it before the
# expensive grid). The stability term is now rollout-MSE against TRUE
# future frames (losses.rollout_mse_loss): identity is maximally
# penalised, and the term directly optimises the per-step MSE the
# stability benchmark measures. The single-step --w-energy 0 is likewise
# set to 0: the energy-drift aux loss is NOT used anywhere in this
# retrain. Stable checkpoints select on val_total (mse + 0.1·rollout-MSE
# — all components identity-repelling); single-step keeps val_mse.
#
# Runs for N = 10, 25, 50, 100 so the stable-vs-single-step contrast can be
# made at every training body count, matching the OOD evaluation grid.
# N=100 stable is included: on the 48 GB RTX 6000 Ada the N=100 GNN stable
# cell fits (tight — see the memory notes and OOM fallback below), where the
# earlier 24 GB GPU OOM'd.
#
# BUGFIX (2026-08-07): the rollout-energy loss previously called the GNN's
# `model.step`, which wraps the state as a W=1 window; the GNN `forward` only
# runs message passing for t in range(1, W), so W=1 ran ZERO message-passing
# rounds. The stability term was optimising a degenerate no-message-passing
# path, which is why the stability-trained GNN's energy drift got WORSE in the
# earlier results. `losses.rollout_energy_loss` now seeds the rollout with the
# true W=5 window and slides it (full message-passing forward), matching the
# evaluation path in evaluate_models.py and stability_benchmark.py. The same
# degenerate-step bug in the OOD runner (real_case_runner.py) was fixed too.
#
# Because the corrected rollout runs the GNN's full W=5 message-passing
# forward K=5 times with no-detach BPTT, its activation memory would be
# ~K x one-forward -- prohibitive at N>=50. Each rollout step's forward is
# therefore gradient-checkpointed inside rollout_energy_loss
# (torch.utils.checkpoint, use_reentrant=False), so only the step inputs are
# retained and activations are recomputed during backward. Peak rollout
# memory is then roughly ONE forward's activations rather than K forwards'.
#
# Controlled experiment: the ONLY scientific difference vs the sweep's
# single-step run (scaling_sweep.py) is --w-rollout 0.1. Same datasets, same
# epochs (MLP=100, LSTM=80, GNN=50), same optimiser defaults, same batch
# sizes (512 / 256 / 128, exactly the sweep's values). The checkpointing is
# an implementation detail of the stable path only (the single-step sweep is
# unchanged) and does not alter the loss being optimised.
#
# Batch sizes: MLP=512, LSTM=256 match the sweep exactly; GNN=128 matches the
# sweep EXCEPT the N=100 stable cell (see the override below). (An earlier
# version halved the MLP to 256 for BPTT memory, which silently confounded
# the stable-vs-single-step contrast with a batch-size change -- audited fix,
# 2026-09-14. The K=5 checkpointed BPTT graph at b=512 still fits the 48 GB
# card with headroom.)
#
# GNN N=100 stable override (2026-09-21, Gemini crosscheck round 2): this
# is the ONE cell that approaches the 48 GB ceiling (~42 GB peak at b=128:
# main forward ~21 GB + one checkpointed rollout recompute ~21 GB) and can
# OOM on fragmentation or a first-iteration spike -- an OOM here would
# waste the ~13 h training run. That cell is therefore trained at b=96
# (~31 GB peak, ~17 GB headroom with expandable_segments; closest batch to
# the b=128 baseline, multiple of 32 for Ada Tensor Core warp scheduling).
# The single-step GNN N=100 run (scaling_sweep.py) is UNTOUCHED and keeps
# b=128 -- only the stable variant pays the BPTT memory cost.
#
# Resumable: a model is skipped if its model_best.pt already exists.
# Run AFTER the main sweep has produced ml_ready_data/N{N}/{mlp,lstm,gnn}/.
#
# Memory: target VM = 48 GB GPU (RTX 6000 Ada, isolated, one model at a time)
# + 16 GB RAM. Datasets are <1 GB in RAM (N=100 GNN ~0.7 GB), so system RAM
# is no issue. GPU peaks (activations + grads, BPTT K=5, rollout checkpointed)
# scale with batch: MLP b=512 ~16 GB, LSTM b=256 ~1 GB, GNN b=128/N=50
# ~11 GB. GNN stable at N=100 is trained at b=96 (~31 GB peak) by default --
# at b=128 it would peak ~42 GB (main forward + one checkpointed rollout
# recompute), which approaches the 48 GB ceiling and risks an OOM that would
# waste the ~13 h cell. If b=96 ever OOMs, drop THAT CELL to 64 (then 32):
#   python gnn_train.py --npz ml_ready_data/N100/gnn/dataset_3d_w5h1s1r.npz \
#     --out training_runs/N100/gnn_stable --epochs 50 --batch-size 64 \
#     --w-energy 0 --w-rollout 0.1 --rollout-K 5

set -euo pipefail

# CUDA allocator: expandable segments let PyTorch map/unmap pages around the
# checkpointed-rollout recompute spikes instead of failing on fragmentation
# when a segment boundary splits a large allocation (Gemini suggestion,
# 2026-09-21). Harmless for every other cell; insurance for the tight GNN
# N=100 stable cell (which trains at b=96, ~31 GB peak vs 48 GB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve the repo root as this script's directory (works when invoked by
# absolute path from anywhere).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Body counts to retrain with rollout loss.
N_VALUES=(10 25 50 100)

declare -A SCRIPT=( [mlp]=mlp_train.py [lstm]=lstm_train.py [gnn]=gnn_train.py )
declare -A EPOCHS=( [mlp]=100 [lstm]=80  [gnn]=50 )
declare -A BATCH=(  [mlp]=512  [lstm]=256 [gnn]=128 )
declare -A ROLLK=(  [mlp]=5    [lstm]=5   [gnn]=5   )

# Failed cells are recorded here and reported at the end: one OOM in a
# 24-model queue must not kill the other 23 (audited fix, 2026-09-14).
FAILED=()

for N in "${N_VALUES[@]}"; do
    # Absolute paths: the trainers (gnn_train.py / lstm_train.py / mlp_train.py)
    # resolve a RELATIVE --out against a doubled repo_root
    # (/root/Universe-Simulation/Universe-Simulation/...), so the checkpoint
    # lands at a doubled path that this script's skip-check (single path) never
    # sees → it would retrain every launch. Absolute --out skips the trainer's
    # prefixing (Path-join with an absolute RHS discards the prefix), so the
    # checkpoint lands exactly where CKPT below points and resumability works.
    OUT_BASE="${REPO}/training_runs/N${N}"
    NPZ_BASE="${REPO}/ml_ready_data/N${N}"
    echo
    echo "============================================================"
    echo "  Stability retraining, N=${N}"
    echo "============================================================"
    for m in mlp lstm gnn; do
        OUT="${OUT_BASE}/${m}_stable"
        NPZ="${NPZ_BASE}/${m}/dataset_3d_w5h1s1r.npz"
        CKPT="${OUT}/model_best.pt"

        if [ -f "${CKPT}" ]; then
            echo "[skip] ${CKPT} already exists"
            continue
        fi
        if [ ! -f "${NPZ}" ]; then
            echo "[error] dataset not found: ${NPZ}"
            echo "        Run the sweep's export step for N=${N} first."
            FAILED+=("N${N}/${m}_stable (no dataset)")
            continue
        fi

        # Per-cell batch override: GNN stable at N=100 drops to b=96
        # (~42 GB peak at b=128 approaches the 48 GB ceiling; see header).
        # Every other cell keeps the sweep-matching batch.
        if [ "${m}" = "gnn" ] && [ "${N}" = "100" ]; then
            B=96
        else
            B=${BATCH[$m]}
        fi

        echo "[train] N=${N} stable ${m}  epochs=${EPOCHS[$m]}  " \
             "batch=${B}  w_rollout=0.1  rollout_K=${ROLLK[$m]}"
        mkdir -p "${OUT}"
        # tee keeps the full stdout for an unattended multi-hour run; the
        # trainer's own exit status survives the pipe via PIPESTATUS.
        if ! python "${SCRIPT[$m]}" \
            --npz        "${NPZ}" \
            --out        "${OUT}" \
            --epochs     "${EPOCHS[$m]}" \
            --batch-size "${B}" \
            --w-energy   0 \
            --w-rollout  0.1 \
            --rollout-K  "${ROLLK[$m]}" 2>&1 | tee "${OUT}/train.log"; then
            echo "[FAIL] N=${N} stable ${m} -- continuing with the queue;" \
                 "re-run this cell later (log: ${OUT}/train.log)"
            FAILED+=("N${N}/${m}_stable (exit ${PIPESTATUS[0]})")
            continue
        fi
        # Post-cell evaluation: same 4-metric table the sweep produces
        # (MSE, |ΔE/E0|, latency, K=50 rollout drift). Output goes to
        # results/N{N}_{m}_stable_metrics.json so the per-cell rollup is
        # available immediately. If eval itself fails (e.g. the trainer
        # finished but the ckpt is corrupt) we record but don't kill the
        # queue — evaluation is recoverable; the training isn't.
        if [ -f "${OUT}/model_best.pt" ]; then
            mkdir -p results
            echo
            echo "[eval] N=${N} stable ${m} -> results/N${N}_${m}_stable_metrics.json"
            if ! python evaluate_models.py \
                --ckpt    "${OUT}/model_best.pt:${m}" \
                --npz     "${NPZ}" \
                --split   test \
                --rollout-K 50 --rollout-batches 4 \
                --json    "results/N${N}_${m}_stable_metrics.json" \
                2>&1 | tee "${OUT}/eval.log"; then
                echo "[FAIL] N=${N} stable ${m} eval failed (training ok); " \
                     "continuing. Log: ${OUT}/eval.log"
                FAILED+=("N${N}/${m}_stable (eval failed)")
            fi
        else
            echo "[warn] N=${N} stable ${m} produced no model_best.pt; " \
                 "skipping eval"
            FAILED+=("N${N}/${m}_stable (no checkpoint)")
        fi
    done
done

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "============================================================"
    echo "  ${#FAILED[@]} cell(s) FAILED -- stable checkpoints missing:"
    for f in "${FAILED[@]}"; do
        echo "    ${f}"
    done
    echo "Re-run this script once the cause is fixed; completed cells"
    echo "are skipped via their existing model_best.pt."
    exit 1
fi
echo "Done. Stable checkpoints under training_runs/N{10,25,50,100}/{mlp,lstm,gnn}_stable/model_best.pt"