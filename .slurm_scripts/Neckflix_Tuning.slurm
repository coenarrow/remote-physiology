#!/bin/bash
#SBATCH --job-name=Neckflix_Tuning
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_Neckflix_Tuning.out
#SBATCH --error=logs/%j_Neckflix_Tuning.err

# What each config in configs/hpc_configs_<size> costs on one A100, and the
# launcher's numbers that follow for the whole node: tools/tuning_report.py
# takes a few real training steps of every model at batch 1, 2, 4, ... until
# the card runs out, and prints the --batch-size, --parallel,
# --nproc-per-node and --num-workers each model's sweep script wants, with
# the hours the sweep takes.
#
#   sbatch .slurm_scripts/Neckflix_Tuning.slurm          # configs/hpc_configs_200
#   sbatch .slurm_scripts/Neckflix_Tuning.slurm 300      # configs/hpc_configs_300
#
# The argument is the frame size, which names the directory of configs. A
# size with no directory yet is costed from another's configs with the
# tool's --resize.
#
# One card and a quarter of the node's cores are enough to measure; GPUS and
# CPUS are the node the sweeps run on, not this allocation. EPOCHS is the
# sweep's, and only scales the hours.
#
# The plan lands in logs/<job id>_tuning_<size>.csv, the measurements it was
# made from beside it in logs/<job id>_tuning_<size>_measured.csv.

GPUS=4
CPUS=48
EPOCHS=20
SIZE="${1:-200}"

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, CPUs per task: ${SLURM_CPUS_PER_TASK}, usable here: $(nproc)"
echo "configs: configs/hpc_configs_${SIZE}"

uv run python tools/tuning_report.py probe --datasets neckflix_hpc \
    --configs "configs/hpc_configs_${SIZE}" \
    --gpus "${GPUS}" --cpus "${CPUS}" --epochs "${EPOCHS}" \
    --out "logs/${SLURM_JOB_ID}_tuning_${SIZE}.csv"
