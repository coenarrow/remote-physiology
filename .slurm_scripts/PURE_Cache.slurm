#!/bin/bash
#SBATCH --job-name=PURE_Cache
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%j_PURE_Cache.out
#SBATCH --error=logs/%j_PURE_Cache.err

# Build the PURE zarr cache on group storage from the raw dataset, then
# validate it. CPU only: no GPU, no `module load cuda`. One store per
# recording, SS-TT.zarr, frames resized to 256x256 (native is 640x480 and
# 27 GB; 256 is about 6 GB and serves every interface in configs/interfaces,
# the largest of which is 144). The cacher's own environment syncs on the
# first run, inside this job. Reruns skip finished stores; add --overwrite
# to the pure-preprocess line to rebuild.
#
#   sbatch .slurm_scripts/PURE_Cache.slurm
#
# PURE_CACHE is the path configs/datasets/pure_hpc.yaml reads; change both
# together. PURE_RAW is the directory holding the SS-TT recording
# directories, each with its PNG frames and its JSON sidecar.

PURE_RAW="/group/pgh004/carrow/data/PURE/PURE"
PURE_CACHE="/group/pgh004/carrow/caches/pure_zarr"

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "raw: ${PURE_RAW} -> cache: ${PURE_CACHE}"

if [ ! -d "${PURE_RAW}" ]; then
    echo "PURE_RAW does not exist: ${PURE_RAW}; edit it at the top of this script" >&2
    exit 1
fi

uv run --project dataset/cachers/pure pure-preprocess --input-dir "${PURE_RAW}" --output-dir "${PURE_CACHE}" --num-workers 4

echo "stores written: $(ls -d "${PURE_CACHE}"/*.zarr | wc -l)"

uv run python tools/validate_cache.py "${PURE_CACHE}"
