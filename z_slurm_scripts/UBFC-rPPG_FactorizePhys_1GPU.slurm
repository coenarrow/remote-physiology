#!/bin/bash
#SBATCH --job-name=UBFC_FactorizePhys_1GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/UBFC_FactorizePhys_1GPU_%j.out
#SBATCH --error=logs/UBFC_FactorizePhys_1GPU_%j.err

module load cuda
cd /mmfs1/data/group/pgh004/carrow/repo/rPPG-Toolbox
mkdir -p logs

PORT=$((29500 + (SLURM_JOB_ID % 1000)))

uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
    main.py --config_file .configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_FACTORIZEPHYS.yaml
