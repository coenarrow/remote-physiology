#!/bin/bash
#SBATCH --job-name=Neckflix_iBVPNet_H200_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_Neckflix_iBVPNet_H200_4GPU.out
#SBATCH --error=logs/%j_Neckflix_iBVPNet_H200_4GPU.err

# iBVPNet on Neckflix at 200 px, 20 epochs a fold, the whole node.
#
#   sbatch .slurm_scripts/Neckflix_iBVPNet_H200_4GPU.slurm            # every participant
#   sbatch .slurm_scripts/Neckflix_iBVPNet_H200_4GPU.slurm 1 2 3      # these, one fold each
#
# With no arguments main.py holds out every participant the dataset config
# admits, in turn. It does not resume, so a sweep longer than the time limit
# is submitted as several jobs, each naming its participants.
#
# Two folds at a time, two A100s each at batch 2 a card, a global batch of 4
# (--parallel 2 --nproc-per-node 2). Measured on one A100 in bfloat16 (job
# 42216): batch 2 peaked at 18.7 GB of 39.5 GB with a step of 0.81 s; batch 4
# fits at 37.1 GB, too close to the card to run a sweep on. The cards set the
# pace, and 4 workers a process (16 in all) keep them fed; job 41489 ran 44
# and was killed for host memory at 128G, so this one asks for 400G of the
# node's 515G. The probe puts the 50 folds at about 22 h, before the records
# each epoch writes and scores.
#
# Each fold writes its own log.txt under runs/${EXPERIMENT}/.

EXPERIMENT="neckflix_hpc_ibvpnet_H200W200"
PARTICIPANTS=("$@")

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

module load cuda

HELD_OUT=()
if [ "${#PARTICIPANTS[@]}" -gt 0 ]; then
    HELD_OUT=(--test-participant-id "${PARTICIPANTS[@]}")
fi

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE}, usable here: $(nproc)"
echo "held out: ${PARTICIPANTS[*]:-every participant}"

uv run python main.py --datasets neckflix_hpc --test-participant-dataset neckflix_hpc \
    "${HELD_OUT[@]}" \
    --config configs/hpc_configs_200/ibvpnet_FS30_W10S1_RGBID_ABP-CVP_H200W200.yaml \
    --experiment "${EXPERIMENT}" \
    --epochs 20 --batch-size 2 --num-workers 4 --parallel 2 --nproc-per-node 2
