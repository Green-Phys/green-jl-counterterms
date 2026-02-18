#!/usr/bin/env python3
SEET_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J seet_it{iter}
#SBATCH -p ludicrous,super,batch
#SBATCH -o seet_it{iter}.o%j
#SBATCH -e seet_it{iter}.e%j
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 48:00:00

export OMP_NUM_THREADS=1
export ScriptDir="/home/umyangyu/install/green-seet/python"
export HDF5_USE_FILE_LOCKING=FALSE
SEETexe="/home/umyangyu/install/green-seet/bin/embedding.exe"
WeakCouplingDir="/pauli-storage/gharsha/NiO"
Solver="/bin/true"


date
echo $SLURM_NODELIST

# Run SEET with external solver
srun -u ${{SEETexe}} \
  --scf_type=GW \
  --BETA {beta}  \
  --itermax 1 \
  --restart {restart} \
  --results_file {results_filename} \
  --weak_results ${{WeakCouplingDir}}/scgw/NiO_GW.h5 \
  --embedding_type SEET  \
  --grid_file {ir_file} \
  --input_file ${{WeakCouplingDir}}/dft/input.h5 \
  --dfintegral_hf_file ${{WeakCouplingDir}}/dft/df_hf_int \
  --dfintegral_file ${{WeakCouplingDir}}/dft/df_int \
  --mixing_type SIGMA_MIXING \
  --mixing_weight {mixing}  \
  --seet_input {transform_filename} \
  --impurity_solver "INCHWORM"  \
  --impurity_solver_exec ${{Solver}}  \
  --impurity_solver_params " "  \
  --dc_data_prefix "dc_int" \
  --seet_root_dir "./"  \
  --spin_symm false \
  --verbose 4

date
"""


IMPURITY_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J imp_it{iter}             # Job name
#SBATCH -p ludicrous,super,batch
#SBATCH -o imp_it{iter}.o%j         # Name of stdout output file
#SBATCH -e imp_it{iter}.e%j         # Name of stderr error file
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH --hint nomultithread
#SBATCH -t 48:00:00

date
echo $SLURM_NODELIST

export OPENBLAS_NUM_THREADS=1
INPUT_FILE={input_file}
PROJECT_DIR=~/src/QMBToolkit/QMBToolkitSOEHYB.jl

# Precompile the package once before launching MPI tasks
# This prevents all MPI ranks from trying to compile simultaneously
julia --check-bounds=no --project="${{PROJECT_DIR}}" -e 'using Pkg; Pkg.instantiate(); using QMBToolkitSOEHYB'

srun julia --check-bounds=no --project=${{PROJECT_DIR}} -e "
using QMBToolkitSOEHYB
QMBToolkitSOEHYB.run(\"${{INPUT_FILE}}\")
"

date
"""

IMPURITY_INPUT_TEMPLATE = """[model]
beta = {beta}
file_path = "./"
hopping_file = "imp_{imp}_hopping.txt"
interaction_file = "imp_{imp}_Uijkl_chem.txt"
hybridization_poles_file = "imp_{imp}_hyb_poles.txt"
hybridization_weights_file = "imp_{imp}_hyb_weights.txt"

[solver.propagator]
is_bold = false
expansion_order = {expansion_order_propagator}

[solver.green]
is_bold = false
expansion_order = {expansion_order_green}
save_linear_size = 1001

[solver.dlr]
rtol = 1e-12
Euv = 10.0

[solver]
output_path = "./"
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import time
import h5py

def submit_slurm_job(script_path: Path) -> str:
    """
    Submit a SLURM job and return the job ID.
    """
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")
    
    # Extract job ID from output (format: "Submitted batch job XXXXX")
    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted job {job_id}: {script_path.name}")
    return job_id

def check_job_status(job_id: str) -> str:
    """
    Check the status of a SLURM job.
    Returns: "COMPLETED", "RUNNING", "PENDING", or "FAILED"
    """
    result = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True)
    if result.returncode != 0:
        return "UNKNOWN"
    
    if "JobState=COMPLETED" in result.stdout:
        return "COMPLETED"
    elif "JobState=RUNNING" in result.stdout:
        return "RUNNING"
    elif "JobState=PENDING" in result.stdout:
        return "PENDING"
    elif "JobState=FAILED" in result.stdout or "JobState=TIMEOUT" in result.stdout:
        return "FAILED"
    else:
        return "UNKNOWN"

def wait_for_job(job_id: str, check_interval: int = 30, max_wait: int = 86400) -> bool:
    """
    Wait for a job to complete.
    Returns True if job completed successfully, False otherwise.
    """
    elapsed = 0
    while elapsed < max_wait:
        status = check_job_status(job_id)
        print(f"Job {job_id} status: {status}")
        
        if status == "COMPLETED":
            print(f"Job {job_id} completed successfully")
            return True
        elif status == "FAILED":
            print(f"Job {job_id} failed")
            return False
        elif status == "UNKNOWN":
            print(f"Job {job_id} not found (may have been purged)")
            return False
        
        time.sleep(check_interval)
        elapsed += check_interval
    
    print(f"Job {job_id} did not complete within {max_wait} seconds")
    return False

def collect_iteration_files(iteration: int, workdir: Path):
    """
    Create a folder named itXXX and move all files matching *itXXX* into it.
    """
    iter_dir = workdir / f"it{iteration}"
    iter_dir.mkdir(exist_ok=True)
    pattern = f"*it{iteration}*"
    for f in workdir.glob(pattern):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))
    # move imp_*.txt to the folder
    for f in workdir.glob(f"imp_*.txt"):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))
    for f in workdir.glob("dc.*result.h5"):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))


def create_impurity_input(iteration: int, imp: int, workdir: Path, args: argparse.Namespace):
    """
    Create input file for impurity solver based on template.
    """
    input_file = workdir / f"imp_{imp}_it{iteration}.toml"
    # 

def process_impurity_input(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    return True

def process_impurity_output(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    return True

def main():
    parser = argparse.ArgumentParser(description="Submit SEET + SOEHYB solver jobs in sequence")
    parser.add_argument("--niters", type=int, default=1, help="Number of SEET iterations to run")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Working directory for job scripts")
    parser.add_argument("--check-interval", type=int, default=5, help="Interval (seconds) to check job status")
    parser.add_argument("--transform_file", type=Path, default="transform.h5", help="Path to transformation file")
    parser.add_argument("--results_file", type=Path, default="seet_results.h5", help="Path to SEET results file")
    parser.add_argument("--mixing", type=float, default=0.4, help="Mixing parameter for self-energy update")
    parser.add_argument("--ir_file", type=Path, default="ir/1e4.h5", help="Path to IR grid file")
    parser.add_argument("--beta", type=float, default=100.0, help="Inverse temperature for IR grid")
    parser.add_argument("--impurity_solver_input_template", type=Path, default="impurity_input_template.toml", help="Template for impurity solver input")
    
    args = parser.parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    ftransform = h5py.File(args.transform_file, 'r')
    nimp = ftransform['nimp'][()]
    job_ids = []
    
    for iteration in range(1, args.niters + 1):
        # ===== SEET job =====
        if iteration == 1:
            restart = "false"
        else:
            restart = "true"
        seet_script_path = workdir / f"seet_it{iteration}.sh"
        seet_script = SEET_SCRIPT_TEMPLATE.format(
            iter=iteration, beta=args.beta, transform_filename=args.transform_file, 
            results_filename=args.results_file, mixing=args.mixing, ir_file=args.ir_file, restart=restart
        )
        seet_script_path.write_text(seet_script)
        seet_script_path.chmod(0o755)
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Submitting SEET job")
        print(f"{'='*60}")
        seet_job_id = submit_slurm_job(seet_script_path)
        job_ids.append(("SEET", iteration, seet_job_id))

        if not wait_for_job(seet_job_id, check_interval=args.check_interval):
            print(f"SEET job failed at iteration {iteration}. Stopping.")
            break

        # ===== Impurity job =====
        for imp in range(nimp):
            create_impurity_input(imp, workdir, args)
            impurity_script_path = workdir / f"imp_{imp}_it{iteration}.sh"
            impurity_script = IMPURITY_SCRIPT_TEMPLATE.format(iter=iteration)
            impurity_script_path.write_text(impurity_script)
            impurity_script_path.chmod(0o755)

        process_impurity_input(iteration, workdir, args)

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Submitting impurity solver job")
        print(f"{'='*60}")
        impurity_job_id = submit_slurm_job(impurity_script_path)
        job_ids.append(("Impurity solver", iteration, impurity_job_id))

        if not wait_for_job(impurity_job_id, check_interval=args.check_interval):
            print(f"Impurity solver job failed at iteration {iteration}. Stopping.")
            collect_iteration_files(iteration, workdir)
            break

        if not process_impurity_output(iteration, workdir, args):
            print(f"Failed to process impurity output at iteration {iteration}. Stopping.")
            collect_iteration_files(iteration, workdir)
            break

        collect_iteration_files(iteration, workdir)
    
    print(f"\n{'='*60}")
    print("Job submission summary:")
    print(f"{'='*60}")
    for job_type, iteration, job_id in job_ids:
        print(f"{job_type:12} Iteration {iteration}: Job ID {job_id}")


if __name__ == "__main__":
    main()
