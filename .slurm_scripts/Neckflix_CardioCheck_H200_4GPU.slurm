#!/bin/bash
#SBATCH --job-name=Neckflix_CardioCheck_H200_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j_Neckflix_CardioCheck_H200_4GPU.out
#SBATCH --error=logs/%j_Neckflix_CardioCheck_H200_4GPU.err

# The cluster check of the two cardio recipes before their LOSO sweeps: each
# model as it ships (configs/hpc_configs_200/cardio*_H200W200.yaml, settled
# on the dev box on 2026-10-14/15, development notes experiment 14) for the
# full 30 epochs on two held-out participants, P033 and P036, then scored.
#
#   sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm                        # both models
#   sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm cardiohydra             # one model
#   PARTICIPANTS="33 34" sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm    # other folds
#
# Why these two folds. P033 is the fold every model was tuned and compared
# on; at 200 px FactorizePhys scores ABP CCC 0.78 there, the best of the
# ported models on any participant in the 30s, and on the dev box
# CardioHydra reached 0.68 at epoch 4 (then slid to 0.62 as the predicted
# scale shrank) and CardioConv 0.38. P036 is the hardest of the five folds
# CardioHydra ran on the dev box: the shape comes (r 0.83) but the pulse
# pressure is 37 mmHg too narrow around a level 8 mmHg high, and no ported
# model but RhythmFormer (0.64) reads it at all. P034 (every ported model
# under 0.1, CardioHydra 0.80) is the alternative.
#
# What "learned pretty well" looks like, from the dev-box folds (10 epochs,
# 128 px cache upsampled to 200, float32, batch 1):
#
#   | fold | CardioHydra ABP CCC / r | CVP CCC / r | CardioConv ABP CCC / r | CVP CCC / r |
#   | P033 | 0.68 / 0.85 (ep 4), 0.62 / 0.89 (ep 8) | 0.40 / 0.61 | 0.38 / 0.58 | 0.22 / 0.37 |
#   | P036 | 0.73 / 0.83 (ep 10)                     | 0.22 / 0.38 | not run     |             |
#
# The cluster recipe differs in what the dev box could not run: the native
# 200 px cache (the masks see 66 px of real detail, not 43), bfloat16, a
# global batch of 4 across two cards, and 30 epochs (4,710 optimiser steps
# against the dev box's 6,280 at batch 1 over 10 epochs; HoldCosine's tail
# sits in the last third of either). On the dev box four of five CardioHydra
# folds rose to the last epoch and P033 slid slowly after epoch 4, so a
# cluster fold that ends near or above the table is the pass; one that
# collapses or sits far below it is the bf16 / DDP / resolution question the
# dev box could not ask. Read the held-out curve with
# tools/tuning_report.py convergence and scripts/evaluate.py, and the masks
# with scripts/visualise.py --checkpoint <fold> --dataset neckflix_hpc
# --test-participant-id <id> --windows 6.
#
# Two folds at once, two A100s each at batch 2 a card (a global batch of 4,
# the batch of every hpc_configs_200 sweep; --parallel 2 --nproc-per-node 2),
# both folds of one model together, the two models one after the other.
# Peak reserved memory a card at rate 4 / wavelet 3, bf16: under 11 GB at
# batch 1 for CardioConv on the dev box, so batch 2 is well inside the
# A100's 39.5 GB. The sweeps ran about 1.5 s a step, loader-bound, 5.5-7 min
# an epoch: 30 epochs about 3-4 h a model, 6-8 h the job. 5 workers a
# process (20 in all), 400G of host memory, as the sweeps.
#
# Neither model has trained on the cluster before, and the dev box cannot
# run bfloat16 (Turing), DistributedDataParallel (no libuv in the Windows
# torch) or Mamba3's Triton kernels under either. Every parameter of both
# models takes a gradient with the training gate open or closed (checked
# 2026-10-14, runs/dev_cardio/exp12_coarse_check/grad_coverage.py), which is
# what DDP needs. A smoke in salloc on two GPUs first, one model each, costs
# minutes and catches anything that would waste the job:
#
#   uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
#       --test-participant-id 33 \
#       --config configs/hpc_configs_200/cardiohydra_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
#       --experiment smoke_cardio_check --epochs 1 --batch-size 2 --limit-windows 8 \
#       --parallel 1 --nproc-per-node 2
#
# The checkout must carry main at d1b0059 or later (coarse-grid masks, the
# coherence background bin, MIXER_* keys, TRACE_INDEPENDENT false for
# CardioConv): `git pull`, then `git log -1` and `git status` before sbatch,
# and leave the checkout alone until the job finishes. Each fold writes its
# own log.txt under runs/${EXPERIMENT_PREFIX}_<model>/; the scoring at the
# end writes evaluation/ beside the folds.

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    MODELS=(cardiohydra cardioconv)
fi
PARTICIPANTS=${PARTICIPANTS:-"33 36"}
EXPERIMENT_PREFIX="neckflix_hpc_check_H200W200_30ep"

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, usable here: $(nproc)"
echo "commit: $(git log -1 --format='%h %s')"
echo "models: ${MODELS[*]}; held out: ${PARTICIPANTS}"

FAILED=()
for model in "${MODELS[@]}"; do
    config="configs/hpc_configs_200/${model}_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml"
    if [ ! -f "${config}" ]; then
        echo "no such model config: ${config}"
        FAILED+=("${model}")
        continue
    fi
    experiment="${EXPERIMENT_PREFIX}_${model}"
    echo "=== ${model}: $(date) ==="
    # shellcheck disable=SC2086
    uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
        --test-participant-id ${PARTICIPANTS} \
        --config "${config}" \
        --experiment "${experiment}" \
        --epochs 30 --batch-size 2 --num-workers 5 --parallel 2 --nproc-per-node 2
    status=$?
    echo "=== ${model} exited ${status}: $(date) ==="
    if [ "${status}" -ne 0 ]; then
        FAILED+=("${model}")
        continue
    fi
    # Score every fold's last epoch (CPU): evaluation/tables/ under the
    # experiment directory, both folds pooled, plus per-fold copies.
    uv run python scripts/evaluate.py "runs/${experiment}"
    for fold in runs/"${experiment}"/*/; do
        uv run python scripts/evaluate.py "${fold}"
    done
    echo "=== ${model} scored: $(date) ==="
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "failed: ${FAILED[*]}"
    exit 1
fi
echo "all ${#MODELS[@]} model(s) done: $(date)"
