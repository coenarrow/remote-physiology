#!/bin/bash
#SBATCH --job-name=Neckflix_RhythmFormer_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_Neckflix_RhythmFormer_4GPU.out
#SBATCH --error=logs/%j_Neckflix_RhythmFormer_4GPU.err

# RhythmFormer on Neckflix at 300 px, every participant held out in turn, 30
# epochs a fold. No --test-participant-id: main.py folds over every
# participant the dataset config admits.
#
#   sbatch .slurm_scripts/Neckflix_RhythmFormer_4GPU.slurm
#
# One fold at a time across all four A100s (--parallel 1 --nproc-per-node 4).
# --batch-size is per process, so 1 on each card is a global batch of 4.
# More cards do not add memory per card; a model that runs a 40 GB A100 out
# of memory at batch 1 needs PRECISION: bfloat16 in its config, or H100s.
#
# The job takes the whole node (--exclusive, and all 48 CPUs by name so the
# task is bound to every one of them), so every CPU on it is dealt to
# the four processes: one for the process itself, the rest its DataLoader
# workers.
#
# Each fold writes its own log.txt under runs/${EXPERIMENT}/.

EXPERIMENT="neckflix_hpc_rhythmformer_H300W300"
GPUS=4

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

WORKERS=$(( SLURM_CPUS_ON_NODE / GPUS - 1 ))
if [ "${WORKERS}" -lt 1 ]; then
    WORKERS=1
fi

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, ${WORKERS} workers for each of ${GPUS} processes"

uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
    --config configs/hpc_configs/rhythmformer_FS30_W10S1_RGBID_ABP-CVP_H300W300.yaml \
    --experiment "${EXPERIMENT}" \
    --epochs 30 --batch-size 1 --num-workers "${WORKERS}" --parallel 1 --nproc-per-node "${GPUS}"
