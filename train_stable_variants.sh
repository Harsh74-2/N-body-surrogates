#!/usr/bin/env bash
# train_stable_variants.sh
# ========================
# Retrain the three surrogates (MLP / LSTM / GNN) with the K-step rollout
# energy loss ENABLED (w_rollout = 0.1): the "stability-trained" variant
# contrasted against the single-step-trained checkpoints in the stability
# benchmark.
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
# sizes (GNN=128 to match the sweep's single-step run). The checkpointing is
# an implementation detail of the stable path only (the single-step sweep is
# unchanged) and does not alter the loss being optimised.
#
# Batch sizes: LSTM and GNN are kept at the SWEEP values (256 / 128) so the
# comparison is clean, only w_rollout differs. MLP is halved (512 -> 256):
# the K=5 no-detach BPTT graph is O(K) in activation memory, and at b=512 the
# MLP BPTT graph is unnecessarily large, so 256 is used.
#
# Resumable: a model is skipped if its model_best.pt already exists.
# Run AFTER the main sweep has produced ml_ready_data/N{N}/{mlp,lstm,gnn}/.
#
# Memory: target VM = 48 GB GPU (RTX 6000 Ada, isolated, one model at a time)
# + 16 GB RAM. Datasets are <1 GB in RAM (N=100 GNN ~0.7 GB), so system RAM
# is no issue. GPU peaks (activations + grads, BPTT K=5, rollout checkpointed):
# MLP b=256 ~8 GB, LSTM b=256 ~1 GB, GNN b=128/N=50 ~11 GB, GNN b=128/N=100
# ~42 GB (main forward ~21 GB + one checkpointed rollout recompute ~21 GB).
# N=10/25/50 fit comfortably; N=100 GNN stable is the tightest cell and can
# approach the 48 GB ceiling. If it OOMs (fragmentation / first-iteration
# spike), fall back to a smaller batch -- peak memory scales with batch, and
# the loss is a w=0.1 regulariser so a smaller batch does not change the
# science:
#   python gnn_train.py --npz ml_ready_data/N100/gnn/dataset_3d_w5h1s1r.npz \
#     --out training_runs/N100/gnn_stable --epochs 50 --batch-size 64 \
#     --w-rollout 0.1 --rollout-K 5

set -euo pipefail

# Resolve the repo root as this script's directory (works when invoked by
# absolute path from anywhere).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Body counts to retrain with rollout loss.
N_VALUES=(10 25 50 100)

declare -A SCRIPT=( [mlp]=mlp_train.py [lstm]=lstm_train.py [gnn]=gnn_train.py )
declare -A EPOCHS=( [mlp]=100 [lstm]=80  [gnn]=50 )
declare -A BATCH=(  [mlp]=256  [lstm]=256 [gnn]=128 )
declare -A ROLLK=(  [mlp]=5    [lstm]=5   [gnn]=5   )

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
            exit 1
        fi

        echo "[train] N=${N} stable ${m}  epochs=${EPOCHS[$m]}  " \
             "batch=${BATCH[$m]}  w_rollout=0.1  rollout_K=${ROLLK[$m]}"
        python "${SCRIPT[$m]}" \
            --npz        "${NPZ}" \
            --out        "${OUT}" \
            --epochs     "${EPOCHS[$m]}" \
            --batch-size "${BATCH[$m]}" \
            --w-rollout  0.1 \
            --rollout-K  "${ROLLK[$m]}"
    done
done

echo
echo "Done. Stable checkpoints under training_runs/N{10,25,50,100}/{mlp,lstm,gnn}_stable/model_best.pt"