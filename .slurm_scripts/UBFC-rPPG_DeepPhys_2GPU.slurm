#!/bin/bash
#SBATCH --job-name=DeepPhys_UBFC_2GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/DeepPhys_UBFC_2GPU_%j.out
#SBATCH --error=logs/DeepPhys_UBFC_2GPU_%j.err

mkdir -p logs

cd "/group/pgh004/carrow/repo/rPPG-Toolbox"

module load cuda

echo "Starting DeepPhys training on UBFC-rPPG (2 GPUs)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"

uv run python -m torch.distributed.run --nproc_per_node=2 \
    main.py --config_file .configs/UBFC-rPPG_UBFC-rPPG_UBFC-rPPG_DEEPPHYS.yaml
