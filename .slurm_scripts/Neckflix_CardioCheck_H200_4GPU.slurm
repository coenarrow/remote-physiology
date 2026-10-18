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

# The cluster check of the cardio recipes before (or beside) their LOSO
# sweeps, and the optional run of the tuning arms. Each arm is a YAML in
# configs/hpc_tuning_p33/ (BASE on the hpc_configs_200 cardio YAML plus one
# change); cardiohydra_baseline and cardioconv_coherent are the two recipes
# as they ship, cardioconv_dense the masked-mean readout they replaced, the
# rest the arms that lost on the dev box (development notes, experiment 14).
# Each arm runs for EPOCHS epochs on every participant in PARTICIPANTS, up
# to four folds at once, one card each, and is scored.
#
#   sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm                              # both shipped recipes, P033 + P036, 20 epochs
#   sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm cardioconv_coherent          # one
#   PARTICIPANTS="33 34 35 36" sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm   # four folds, one round
#   EPOCHS=10 sbatch .slurm_scripts/Neckflix_CardioCheck_H200_4GPU.slurm cardiohydra_tv0.02 cardiohydra_abpamp   # tuning arms
#
# The first run of this script (job 151998, 2026-10-18, 30 epochs, native
# 650 px cache, two folds at a time over two cards each) is the reference:
#
#   | fold | CardioHydra ABP CCC / r, epoch 30 (best) | CVP CCC / r | CardioConv dense ABP CCC / r | CVP CCC / r |
#   | P033 | 0.65 / 0.92 (0.71 at epoch 15)           | 0.40 / 0.53 | 0.22 / 0.43                  | 0.22 / 0.35 |
#   | P036 | 0.74 / 0.84 (0.77 at epoch 22)           | 0.18 / 0.34 | 0.21 / 0.36                  | 0.19 / 0.29 |
#
# which reproduced the dev box almost number for number (experiment 14's
# addendum): bf16, DDP and the native cache changed nothing, the held-out
# peak came by epoch 15-22 and the second half of the schedule transferred
# nothing, hence 20 epochs here and in the sweeps. CardioConv with the
# masked-mean readout did not fit the ABP shape even on the training set
# (its training CCC term never left 0.62); COHERENT_POOLING (experiment 16)
# took the P033 fold to 0.59-0.64 on the dev box and is what ships, so a
# cardioconv_coherent fold near or above that is the pass and
# cardioconv_dense is the ablation.
#
# That job also ran at 3.2 s a step against the ported sweeps' 1.45 s, the
# loader resizing 650 px windows in a dense einsum. The cache is now 200 px
# (configs/datasets/neckflix_hpc.yaml), the loader resizes nothing, and the
# layout is the sweeps': one process and one card a fold at batch 4
# (--parallel 4 --nproc-per-node 1), 10 workers a fold. Read the first
# epoch's step time in each fold's log.txt; the card alone should be well
# under 1 s. Score the held-out curve per epoch with
# runs/dev_cardio/exp14_final_tuning/score_folds.py (scratch copies), the
# fold with scripts/evaluate.py, and the masks with scripts/visualise.py
# --checkpoint <fold> --dataset neckflix_hpc --test-participant-id <id>.
#
# The checkout must carry the coherent-pooling commit of 2026-10-18 or
# later; `git pull`, `uv sync` (the lock gained opencv-python-headless),
# then `git log -1` and `git status` before sbatch, and leave the checkout
# alone until the job finishes. A smoke in salloc on one GPU first:
#
#   uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
#       --test-participant-id 33 \
#       --config configs/hpc_tuning_p33/cardioconv_coherent.yaml \
#       --experiment smoke_cardio_check --epochs 1 --batch-size 4 --limit-windows 8
#
# Each fold writes its own log.txt under runs/neckflix_hpc_check_<arm>/; the
# scoring at the end writes evaluation/ beside the folds and inside each.

ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
    ARMS=(cardiohydra_baseline cardioconv_coherent)
fi
PARTICIPANTS=${PARTICIPANTS:-"33 36"}
EPOCHS=${EPOCHS:-20}
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
        --epochs "${EPOCHS}" --batch-size 4 --num-workers 10 --parallel 4 --nproc-per-node 1
    status=$?
    echo "=== arm ${arm} exited ${status}: $(date) ==="
    if [ "${status}" -ne 0 ]; then
        FAILED+=("${arm}")
        continue
    fi
    # Score every fold's last epoch (CPU): evaluation/tables/ under the
    # experiment directory, the folds pooled, plus a copy inside each fold
    # (the pooled evaluation/ directory is not a fold and is skipped).
    uv run python scripts/evaluate.py "runs/${experiment}"
    for fold in runs/"${experiment}"/*/; do
        [ -f "${fold}/config.yaml" ] || continue
        uv run python scripts/evaluate.py "${fold}"
    done
    echo "=== arm ${arm} scored: $(date) ==="
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "failed arms: ${FAILED[*]}"
    exit 1
fi
echo "all ${#ARMS[@]} arm(s) done: $(date)"
