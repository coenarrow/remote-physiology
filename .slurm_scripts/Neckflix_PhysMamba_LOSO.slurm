#!/bin/bash
#SBATCH --job-name=Neckflix_PhysMamba_LOSO
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --array=1-40%8
#SBATCH --output=logs/%A_%a_Neckflix_PhysMamba_LOSO.out
#SBATCH --error=logs/%A_%a_Neckflix_PhysMamba_LOSO.err

# Leave-one-subject-out PhysMamba on Neckflix: one array task per participant.
#
# Build the fold list first (metadata-only, safe on a login node), then size the
# array to it:
#   uv run python tools/list_neckflix_folds.py \
#       --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --prefix P > folds.txt
#   sbatch --array=1-$(wc -l < folds.txt)%8 .slurm_scripts/Neckflix_PhysMamba_LOSO.slurm

mkdir -p logs

cd "/group/pgh004/carrow/repo/rPPG-Toolbox"

module load cuda

FOLDS=folds.txt
PARTICIPANT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${FOLDS}")
if [ -z "${PARTICIPANT}" ]; then
    echo "No participant on line ${SLURM_ARRAY_TASK_ID} of ${FOLDS}" >&2
    exit 1
fi

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: holding out ${PARTICIPANT}"
echo "Node: ${SLURMD_NODENAME}"

uv run python neckflix_main.py \
    --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml \
    --test_participants "${PARTICIPANT}" \
    --num_workers ${SLURM_CPUS_PER_TASK}
