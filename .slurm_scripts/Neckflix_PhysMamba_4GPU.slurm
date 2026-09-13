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

# PhysMamba on Neckflix, each fold across all 4 GPUs: one job, one fold at
# a time under torch.distributed.run with one process per GPU, on a port
# derived from the job id. The participants to hold out are this script's
# arguments; with none, every participant the dataset admits is held out in
# turn. Each fold is one scripts/run.py run with its own directory and
# log.txt under runs/neckflix_physmamba/. For small interfaces, four folds
# at a time on one GPU each (--parallel 4 --nproc-per-node 1) does the same
# work without DDP; --gres stays --parallel times --nproc-per-node.
#
#   sbatch .slurm_scripts/Neckflix_PhysMamba_4GPU.slurm 32 33 34

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}, holding out ${*:-every participant}"

FOLDS=()
if [ "$#" -gt 0 ]; then
    FOLDS=(--test-participant-id "$@")
fi

uv run python main.py --datasets neckflix --test-participant-dataset neckflix "${FOLDS[@]}" --model physmamba --interface configs/interfaces/interface_neckflix.yaml --training configs/training/physmamba_training.yaml --parallel 1 --nproc-per-node 4
