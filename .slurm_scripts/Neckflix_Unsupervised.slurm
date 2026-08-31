#!/bin/bash
#SBATCH --job-name=Neckflix_Unsupervised
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_Neckflix_Unsupervised.out
#SBATCH --error=logs/%j_Neckflix_Unsupervised.err

# All seven traditional rPPG methods over the Neckflix zarr cache, scored
# separately against every trace each recording carries (ABP / CVP / ECG).
# CPU only: no method here touches a GPU.

mkdir -p logs

cd "/group/pgh004/carrow/repo/rPPG-Toolbox"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"

uv run python main.py \
    --config_file configs/neckflix/NECKFLIX_UNSUPERVISED.yaml \
    --num_workers ${SLURM_CPUS_PER_TASK}
