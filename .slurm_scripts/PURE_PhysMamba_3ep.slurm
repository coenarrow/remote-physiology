#!/bin/bash
#SBATCH --job-name=PURE_PhysMamba_3ep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_PURE_PhysMamba_3ep.out
#SBATCH --error=logs/%j_PURE_PhysMamba_3ep.err

# PhysMamba on PURE, three epochs, every participant held out in turn: the
# HPC twin of the dev-box test run. Two folds at a time, one GPU each, so
# the recipe (batch 4 per process) means exactly what it means locally.
# The participants to hold out are this script's arguments; with none,
# every participant the cache admits. Each fold is a scripts/run.py run
# with its own directory and log.txt under runs/.
#
#   sbatch .slurm_scripts/PURE_PhysMamba_3ep.slurm
#   sbatch .slurm_scripts/PURE_PhysMamba_3ep.slurm 01 02
#
# Sizing: --cpus-per-task is --parallel times the recipe's NUM_WORKERS (4);
# --gres is --parallel times --nproc-per-node.

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}, holding out ${*:-every participant}"

FOLDS=()
if [ "$#" -gt 0 ]; then
    FOLDS=(--test-participant-id "$@")
fi

uv run python main.py --datasets pure_hpc --test-participant-dataset pure_hpc "${FOLDS[@]}" --model physmamba --interface configs/interfaces/physmamba_interface.yaml --training configs/training/physmamba_3ep.yaml --parallel 2 --nproc-per-node 1
