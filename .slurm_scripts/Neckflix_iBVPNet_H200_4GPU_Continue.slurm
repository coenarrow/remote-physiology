#!/bin/bash
#SBATCH --job-name=Neckflix_iBVPNet_H200_4GPU_Continue
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j_Neckflix_iBVPNet_H200_4GPU_Continue.out
#SBATCH --error=logs/%j_Neckflix_iBVPNet_H200_4GPU_Continue.err

# iBVPNet on Neckflix at 200 px: 10 more epochs on every fold of the
# 20-epoch sweep, bringing each fold to 30.
#
#   sbatch .slurm_scripts/Neckflix_iBVPNet_H200_4GPU_Continue.slurm          # every fold
#   sbatch .slurm_scripts/Neckflix_iBVPNet_H200_4GPU_Continue.slurm 1 2 3    # these folds
#
# main.py --init-from continues each fold directory of runs/${SOURCE}/ from
# its latest epoch (weights only: the checkpoints hold no optimiser state)
# with a fresh OneCycle at the recipe's LR over the 10 new epochs, numbered
# 21-30 in losses.csv and epoch_NN/. The new folds keep their names with a
# new stamp under runs/${SOURCE}_continued/, a directory of their own so the
# evaluation never pools a participant twice. The checkout must carry the
# --init-from launcher: git pull before sbatch.
#
# Same layout as Neckflix_iBVPNet_H200_4GPU.slurm (two folds at a time, two
# A100s each, batch 2 a card). That sweep's 20 epochs took about two days,
# so 10 fit in 2-00:00:00 with margin.

SOURCE="neckflix_hpc_ibvpnet_H200W200"
PARTICIPANTS=("$@")

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

FOLDS=()
if [ "${#PARTICIPANTS[@]}" -gt 0 ]; then
    FOLDS=(--test-participant-id "${PARTICIPANTS[@]}")
fi

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, usable here: $(nproc)"
echo "continuing: ${PARTICIPANTS[*]:-every fold} of runs/${SOURCE}"

uv run python main.py --init-from "runs/${SOURCE}" \
    "${FOLDS[@]}" \
    --experiment "${SOURCE}_continued" \
    --epochs 10 --batch-size 2 --num-workers 4 --parallel 2 --nproc-per-node 2
