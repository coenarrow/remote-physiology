#!/bin/bash
#SBATCH --job-name=Neckflix_Cache
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_Neckflix_Cache.out
#SBATCH --error=logs/%j_Neckflix_Cache.err

# Build the Neckflix zarr cache on group storage from the raw dataset, then
# validate it. CPU only: no GPU, no `module load cuda`. Native resolution,
# every modality and every perspective including the event camera. The
# cacher's own environment syncs on the first run, inside this job. Reruns
# skip finished stores; add --overwrite to the neckflix-preprocess line to
# rebuild.
#
#   sbatch .slurm_scripts/Neckflix_Cache.slurm
#
# The event camera needs the ECF HDF5 codec plugin. The job builds it on
# its first run and skips the build once dataset/cachers/neckflix/hdf5_plugin
# holds it. The hdf5 module's shared library is unreadable and its CMake
# config is broken, so the build bypasses the config and links the static
# library; the plugin then needs no module at runtime.
#
# Memory: the cacher peaks near 12 GB per worker on the largest event
# recordings, so --mem is at least 12G times --num-workers.
#
# CACHE_DIR is the path configs/datasets/neckflix_hpc.yaml reads; change
# both together. RAW_DIR is the dataset root containing data/P*/.

RAW_DIR="/group/pgh004/carrow/data/NECKFLIX"
CACHE_DIR="/group/pgh004/carrow/caches/neckflix_zarr"
CACHER="dataset/cachers/neckflix"

mkdir -p logs

cd "/group/pgh004/carrow/repo/remote-physiology"

echo "Job ID: ${SLURM_JOB_ID}, node ${SLURMD_NODENAME}"
echo "raw: ${RAW_DIR} -> cache: ${CACHE_DIR}"

if [ ! -d "${RAW_DIR}" ]; then
    echo "RAW_DIR does not exist: ${RAW_DIR}; edit it at the top of this script" >&2
    exit 1
fi

if [ ! -f "${CACHER}/pyproject.toml" ]; then
    echo "${CACHER} is empty; run: git submodule update --init ${CACHER}" >&2
    exit 1
fi

PLUGIN_DIR="${PWD}/${CACHER}/hdf5_plugin"
if [ ! -f "${PLUGIN_DIR}/libH5Zecf.so" ]; then
    echo "building the ECF plugin into ${PLUGIN_DIR}"
    module load gcc hdf5 cmake
    BUILD_DIR="${CACHER}/plugin/build"
    cmake -S "${CACHER}/plugin" -B "${BUILD_DIR}" \
        -DCMAKE_INSTALL_PREFIX="${PLUGIN_DIR}" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON \
        -DHDF5_ROOT=/uwahpc/rocky9/devel/gcc/14.3.0/hdf5/1.14.6 || exit 1
    cmake --build "${BUILD_DIR}" -j "${SLURM_CPUS_PER_TASK}" || exit 1
    cmake --install "${BUILD_DIR}" || exit 1
    # The plugin loader wants the shared libraries directly under the prefix.
    cp -a "${PLUGIN_DIR}"/lib/hdf5/plugin/* "${PLUGIN_DIR}"/
    cp -a "${PLUGIN_DIR}"/lib/libhdf5_ecf_codec* "${PLUGIN_DIR}"/
    rm -rf "${PLUGIN_DIR}"/lib "${PLUGIN_DIR}"/include "${PLUGIN_DIR}"/share "${BUILD_DIR}"
    if [ ! -f "${PLUGIN_DIR}/libH5Zecf.so" ]; then
        echo "ECF plugin build produced no libH5Zecf.so in ${PLUGIN_DIR}" >&2
        exit 1
    fi
fi
export HDF5_PLUGIN_PATH="${PLUGIN_DIR}"
echo "ECF plugin: ${HDF5_PLUGIN_PATH}"

uv run --project "${CACHER}" neckflix-preprocess --input-dir "${RAW_DIR}" --output-dir "${CACHE_DIR}" --num-workers 4

echo "stores written: $(ls -d "${CACHE_DIR}"/*.zarr | wc -l)"

uv run python tools/validate_cache.py "${CACHE_DIR}"
