#!/bin/bash
#SBATCH --job-name=PURE_PhysMamba_30ep
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j_PURE_PhysMamba_30ep.out
#SBATCH --error=logs/%j_PURE_PhysMamba_30ep.err

# PhysMamba on PURE, thirty epochs, every participant held out in turn. Two
# folds at a time, each across two GPUs: batch 2 per rank, global batch 4.
# The participants to hold out are this script's arguments; with none,
# every participant the cache admits. Each fold is a scripts/run.py run
# with its own directory and log.txt under runs/.
#
#   sbatch .slurm_scripts/PURE_PhysMamba_30ep.slurm
#   sbatch .slurm_scripts/PURE_PhysMamba_30ep.slurm 01 02
#
# Sizing: --gres is --parallel times --nproc-per-node; --cpus-per-task is
# that product times --num-workers; --mem allows about 1 GB per worker,
# doubled at the test pass, plus each rank's CUDA context. Batch 2 per rank
# already reserves 97% of an A100 at this interface; batch 4 does not fit.

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}, holding out ${*:-every participant}"

FOLDS=()
if [ "$#" -gt 0 ]; then
    FOLDS=(--test-participant-id "$@")
fi

uv run python main.py --datasets pure_hpc --test-participant-dataset pure_hpc "${FOLDS[@]}" \
    --config configs/experiment_configs/physmamba_FS30_W10S1_RGB_PPG_H256W256.yaml \
    --epochs 30 --batch-size 2 --num-workers 2 --parallel 2 --nproc-per-node 2
