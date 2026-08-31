#!/bin/bash
#SBATCH --job-name=Neckflix_PhysMamba_4GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_Neckflix_PhysMamba_4GPU.out
#SBATCH --error=logs/%j_Neckflix_PhysMamba_4GPU.err

# One PhysMamba fold across 4 GPUs. Only the training split is sharded; the
# test split runs on rank 0 so per-signal metrics are computed once.

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

PARTICIPANT="${1:-P015}"

echo "Job ID: ${SLURM_JOB_ID}, holding out ${PARTICIPANT}"
echo "Node: ${SLURMD_NODENAME}"

uv run python -m torch.distributed.run --nproc_per_node=4 \
    main.py \
    --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml \
    --test_participants "${PARTICIPANT}" \
    --num_workers 4
