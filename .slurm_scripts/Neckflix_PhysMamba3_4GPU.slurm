#!/bin/bash
#SBATCH --job-name=Neckflix_PhysMamba3_4GPU
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_Neckflix_PhysMamba3_4GPU.out
#SBATCH --error=logs/%j_Neckflix_PhysMamba3_4GPU.err

# PhysMamba3 on Neckflix at 300 px, 20 epochs a fold, the whole node.
#
#   sbatch .slurm_scripts/Neckflix_PhysMamba3_4GPU.slurm            # every participant
#   sbatch .slurm_scripts/Neckflix_PhysMamba3_4GPU.slurm 1 2 3      # these, one fold each
#
# With no arguments main.py holds out every participant the dataset config
# admits, in turn. It does not resume, so a sweep longer than the time limit
# is submitted as several jobs, each naming its participants.
#
# One fold at a time across all four A100s at batch 1 a card, a global batch
# of 4 (--parallel 1 --nproc-per-node 4). Measured on one A100 in bfloat16
# (job 42033): batch 1 peaked at 29.3 GB of 39.5 GB with a step of 2.25 s;
# batch 2 ran out. Job 41489 ran 11 workers a process and was killed for host
# memory at 128G; 4 a process is what the card and the test pass consume.
#
# Each fold writes its own log.txt under runs/${EXPERIMENT}/.

EXPERIMENT="neckflix_hpc_physmamba3_H300W300"
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
    --config configs/hpc_configs_300/physmamba3_FS30_W10S1_RGBID_ABP-CVP_H300W300.yaml \
    --experiment "${EXPERIMENT}" \
    --epochs 20 --batch-size 1 --num-workers 4 --parallel 1 --nproc-per-node 4
