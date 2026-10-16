#!/bin/bash
#SBATCH --job-name=Neckflix_CardioCheck_H200_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j_Neckflix_CardioCheck_H200_4GPU.out
#SBATCH --error=logs/%j_Neckflix_CardioCheck_H200_4GPU.err

# The cluster check of the cardio recipes before their LOSO sweeps, and the
# optional re-run of the tuning arms on the native cache. Each arm is a YAML
# in configs/hpc_tuning_p33/ (BASE on the hpc_configs_200 cardio YAML plus
# one change); cardiohydra_baseline and cardioconv_dense are the two
# recipes as they ship, the rest the arms that lost on the dev box
# (development notes, experiment 14). Each arm runs for EPOCHS epochs on
# every participant in PARTICIPANTS, two folds at once, and is scored.
#
#   sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm                              # both shipped recipes, P033 + P036, 30 epochs
#   sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm cardiohydra_baseline         # one
#   PARTICIPANTS="33 34" sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm         # other folds
#   EPOCHS=10 sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm cardiohydra_tv0.02 cardiohydra_mamba0.3   # tuning arms
#
# Why P033 and P036. P033 is the fold every model was tuned and compared
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
#   | fold | CardioHydra ABP CCC / r                | CVP CCC / r | CardioConv ABP CCC / r | CVP CCC / r |
#   | P033 | 0.68 / 0.85 (ep 4), 0.62 / 0.89 (ep 8) | 0.40 / 0.61 | 0.38 / 0.58            | 0.22 / 0.37 |
#   | P036 | 0.73 / 0.83 (ep 10)                    | 0.22 / 0.38 | not run                |             |
#
# The cluster recipe differs in what the dev box could not run: the native
# 200 px cache (the masks see 66 px of real detail, not 43), bfloat16, a
# global batch of 4 across two cards, and 30 epochs (4,710 optimiser steps
# against the dev box's 6,280 at batch 1 over 10 epochs; HoldCosine's tail
# sits in the last third of either). On the dev box four of five CardioHydra
# folds rose to the last epoch and P033 slid slowly after epoch 4, so a
# cluster fold that ends near or above the table is the pass; one that
# collapses or sits far below it is the bf16 / DDP / resolution question the
# dev box could not ask. The tuning arms are read against the shipped
# recipe's fold the same way; on the dev box every one scored below it, so
# an arm that wins here is a resolution effect worth knowing about. Read the
# held-out curve with tools/tuning_report.py convergence and
# scripts/evaluate.py, and the masks with scripts/visualise.py --checkpoint
# <fold> --dataset neckflix_hpc --test-participant-id <id> --windows 6.
#
# Two folds at once, two A100s each at batch 2 a card (a global batch of 4,
# the batch of every hpc_configs_200 sweep; --parallel 2 --nproc-per-node 2),
# one arm after another. Peak reserved memory a card at rate 4 / wavelet 3,
# bf16: under 11 GB at batch 1 for CardioConv on the dev box, so batch 2 is
# well inside the A100's 39.5 GB. The sweeps ran about 1.5 s a step,
# loader-bound, 5.5-7 min an epoch: 30 epochs about 3-4 h an arm of two
# folds, 6-8 h for the two shipped recipes. 5 workers a process (20 in all),
# 400G of host memory, as the sweeps. An arm that fails is reported and the
# next one starts.
#
# Neither model has trained on the cluster before, and the dev box cannot
# run bfloat16 (Turing), DistributedDataParallel (no libuv in the Windows
# torch) or Mamba3's Triton kernels under either (the same mamba-ssm 2.3.2
# install ran PhysMamba's sweep). Every parameter of both models takes a
# gradient with the training gate open or closed (checked 2026-10-14,
# runs/dev_cardio/exp12_coarse_check/grad_coverage.py), which is what DDP
# needs. A smoke in salloc on two GPUs first, one model each, costs minutes
# and catches anything that would waste the job:
#
#   uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
#       --test-participant-id 33 \
#       --config configs/hpc_tuning_p33/cardiohydra_baseline.yaml \
#       --experiment smoke_cardio_check --epochs 1 --batch-size 2 --limit-windows 8 \
#       --parallel 1 --nproc-per-node 2
#
# The checkout must carry main at d1b0059 or later (coarse-grid masks, the
# coherence background bin, MIXER_* keys, TRACE_INDEPENDENT false for
# CardioConv): `git pull`, then `git log -1` and `git status` before sbatch,
# and leave the checkout alone until the job finishes. Each fold writes its
# own log.txt under runs/neckflix_hpc_check_<arm>/; the scoring at the end
# writes evaluation/ beside the folds and inside each.

ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
    ARMS=(cardiohydra_baseline cardioconv_dense)
fi
PARTICIPANTS=${PARTICIPANTS:-"33 36"}
EPOCHS=${EPOCHS:-30}
CONFIG_DIR="configs/hpc_tuning_p33"

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, usable here: $(nproc)"
echo "commit: $(git log -1 --format='%h %s')"
echo "arms: ${ARMS[*]}; held out: ${PARTICIPANTS}; epochs: ${EPOCHS}"

FAILED=()
for arm in "${ARMS[@]}"; do
    config="${CONFIG_DIR}/${arm}.yaml"
    if [ ! -f "${config}" ]; then
        echo "no such arm: ${config}"
        FAILED+=("${arm}")
        continue
    fi
    experiment="neckflix_hpc_check_${arm}"
    echo "=== arm ${arm}: $(date) ==="
    # shellcheck disable=SC2086
    uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
        --test-participant-id ${PARTICIPANTS} \
        --config "${config}" \
        --experiment "${experiment}" \
        --epochs "${EPOCHS}" --batch-size 2 --num-workers 5 --parallel 2 --nproc-per-node 2
    status=$?
    echo "=== arm ${arm} exited ${status}: $(date) ==="
    if [ "${status}" -ne 0 ]; then
        FAILED+=("${arm}")
        continue
    fi
    # Score every fold's last epoch (CPU): evaluation/tables/ under the
    # experiment directory, the folds pooled, plus a copy inside each fold.
    uv run python scripts/evaluate.py "runs/${experiment}"
    for fold in runs/"${experiment}"/*/; do
        uv run python scripts/evaluate.py "${fold}"
    done
    echo "=== arm ${arm} scored: $(date) ==="
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "failed arms: ${FAILED[*]}"
    exit 1
fi
echo "all ${#ARMS[@]} arm(s) done: $(date)"
