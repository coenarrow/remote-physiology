#!/bin/bash
#SBATCH --job-name=Neckflix_PhysMamba_2GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_Neckflix_PhysMamba_2GPU.out
#SBATCH --error=logs/%j_Neckflix_PhysMamba_2GPU.err

# PhysMamba on Neckflix at 300 px, every participant held out in turn, 30
# epochs a fold. No --test-participant-id: main.py folds over every
# participant the dataset config admits.
#
#   sbatch .slurm_scripts/Neckflix_PhysMamba_2GPU.slurm
#
# One fold at a time across both A100s (--parallel 1 --nproc-per-node 2).
# --batch-size is per process, so 1 on each card is a global batch of 2.
# Batch 2 a card in float32 ran the 40 GB A100 out of memory at 300 px, so
# the config trains in bfloat16 and each card takes one window.
#
# Each fold writes its own log.txt under runs/${EXPERIMENT}/.

EXPERIMENT="neckflix_hpc_physmamba_H300W300"

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"

uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
    --config configs/hpc_configs_300/physmamba_FS30_W10S1_RGBID_ABP-CVP_H300W300.yaml \
    --experiment "neckflix_hpc_physmamba_H300W300" \
    --epochs 30 --batch-size 1 --num-workers 4 --parallel 1 --nproc-per-node 2
