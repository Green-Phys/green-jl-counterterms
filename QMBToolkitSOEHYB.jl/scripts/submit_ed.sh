#!/bin/bash
#-------------------------------------------------------------
#SBATCH -J ed             # Job name
#SBATCH -o ed.o%j         # Name of stdout output file
#SBATCH -e ed.e%j         # Name of stderr error file
#SBATCH -p ludicrous,super,batch
#SBATCH --qos special
#SBATCH --exclusive
#SBATCH --hint nomultithread
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 48:00:00

date

export OPENBLAS_NUM_THREADS=1

# Get the project directory
PROJECT_DIR=~/src/QMBToolkit/QMBToolkitSOEHYB.jl

#=============================================================================
# ED Green's function parameters - modify these as needed
#=============================================================================
HOPPING_FILE="./hopping.txt"
INTERACTION_FILE="./Uijkl.txt"
OUTPUT_FILE="./ed_green.h5"
BETA=10.0
DLR_RTOL=1e-14
DLR_EUV=500.0
N_TAU=1001
N_FLAVOR=                # Leave empty for all flavors, or set to integer
SPIN_FILTER=true         # true or false

#=============================================================================
# Build command line arguments
#=============================================================================
ARGS="--hopping=${HOPPING_FILE}"
ARGS="${ARGS} --interaction=${INTERACTION_FILE}"
ARGS="${ARGS} --output=${OUTPUT_FILE}"
ARGS="${ARGS} --beta=${BETA}"
ARGS="${ARGS} --dlr_rtol=${DLR_RTOL}"
ARGS="${ARGS} --dlr_Euv=${DLR_EUV}"
ARGS="${ARGS} --n_tau=${N_TAU}"

if [ -n "${N_FLAVOR}" ]; then
    ARGS="${ARGS} --n_flavor=${N_FLAVOR}"
fi

if [ "${SPIN_FILTER}" = "false" ]; then
    ARGS="${ARGS} --no_spin_filter"
fi

#=============================================================================
# Print job info
#=============================================================================
echo "=== Job Parameters ==="
echo "Hopping file:     ${HOPPING_FILE}"
echo "Interaction file: ${INTERACTION_FILE}"
echo "Output file:      ${OUTPUT_FILE}"
echo "Beta:             ${BETA}"
echo "DLR rtol:         ${DLR_RTOL}"
echo "DLR Euv:          ${DLR_EUV}"
echo "N tau:            ${N_TAU}"
echo "N flavor:         ${N_FLAVOR:-all}"
echo "Spin filter:      ${SPIN_FILTER}"
echo "Nodes:            ${SLURM_NNODES}"
echo "Tasks:            ${SLURM_NTASKS}"
echo "======================"

#=============================================================================
# Precompile and run
#=============================================================================
julia --check-bounds=no --project="${PROJECT_DIR}" -e '
    using Pkg; Pkg.instantiate()
    using MPI, QMBToolkitED, QMBToolkitDLR, HDF5
'

srun julia --check-bounds=no --project=${PROJECT_DIR} \
    ${PROJECT_DIR}/scripts/ed_green.jl ${ARGS}

date
