#!/usr/bin/env python3
SEET_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J seet_it_{iteration}
#SBATCH -p ludicrous,super,batch
#SBATCH -o seet_it_{iteration}.o%j
#SBATCH -e seet_it_{iteration}.e%j
#SBATCH --exclusive
#SBATCH --qos special
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH -t 48:00:00

export OMP_NUM_THREADS=1
export ScriptDir="/home/umyangyu/install/green-seet/python"
export HDF5_USE_FILE_LOCKING=FALSE
SEETexe="/home/umyangyu/install/green-seet/bin/embedding.exe"
WeakCouplingDir="{weak_coupling_dir}"
Solver="/bin/true"


date
echo $SLURM_NODELIST

# Run SEET with external solver
srun -u $SEETexe \
  --scf_type=GW \
  --BETA {beta} \
  --itermax 1 \
  --restart {restart} \
  --results_file {results_filename} \
  --weak_results $WeakCouplingDir/scgw/NiO_GW_1000_lastiter.h5 \
  --embedding_type SEET \
  --grid_file {ir_file} \
  --input_file $WeakCouplingDir/dft/input.h5 \
  --dfintegral_hf_file $WeakCouplingDir/dft/df_hf_int \
  --dfintegral_file $WeakCouplingDir/dft/df_int \
  --mixing_type SIGMA_MIXING \
  --mixing_weight {mixing} \
  --seet_input {transform_filename} \
  --impurity_solver "INCHWORM" \
  --impurity_solver_exec $Solver \
  --impurity_solver_params " " \
  --dc_data_prefix "../dc_int" \
  --seet_root_dir "./" \
  --spin_symm false \
  --verbose 4

date
"""


IMPURITY_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J imp_{imp}_it_{iteration}             # Job name
#SBATCH -p super
#SBATCH -o imp_{imp}_it_{iteration}.o%j         # Name of stdout output file
#SBATCH -e imp_{imp}_it_{iteration}.e%j         # Name of stderr error file
#SBATCH --exclusive
#SBATCH --qos special
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --hint nomultithread
#SBATCH -t 48:00:00

date
echo $SLURM_NODELIST

export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH=$HOME/install/arpack/lib64:$LD_LIBRARY_PATH
SolversDir="/home/umyangyu/install/seet_solvers/"

NBATH_PER_ORBITAL={n_bath_per_orbital}             
BETA={beta}                    
NORBITALS={n_orbital_impurity}                  
NBATH_TOTAL=$((NORBITALS * NBATH_PER_ORBITAL))  
NSITES=$((NORBITALS + NBATH_TOTAL)) 
INPUT_FILE="./ed.{imp}.input.h5"
OUTPUT_FILE="./ed.{imp}.result.h5"

srun $SolversDir/bin/ed_solver \
  --arpack.NEV=8 \
  --arpack.NCV=25 \
  --lanc.NOMEGA=1000 \
  --FREQ_FILE={ir_file} \
  --FREQ_PATH=/fermi/ngrid \
  --NSITES=$NSITES \
  --NSPINS={n_spin_impurity} \
  --INPUT_FILE=$INPUT_FILE \
  --OUTPUT_FILE=$OUTPUT_FILE \
  --arpack.SECTOR=false \
  --siam.NORBITALS=$NORBITALS \
  --spinstorage.ORBITAL_NUMBER=$NORBITALS \
  --lanc.BETA=$BETA

date
"""


IMPURITY_PREPROCESSING_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -p super,batch
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH -J ed_gen_it_{iteration}
#SBATCH -t 1:00:00
#SBATCH -o ed_gen_it_{iteration}.o%j
#SBATCH -e ed_gen_it_{iteration}.e%j
#SBATCH --qos special

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Running on $(hostname) at $(date)"

srun -u /home/umyangyu/install/ed_input_generator/bin/ed_input_generator \
  --BETA {beta} \
  --grid_file {ir_file} \
  --input_file {input_file} \
  --seet_input {transform_file} \
  --bath_file {bath_file} \
  --seet_root_dir ./ \
  --spin_symm false \
  --results_file {results_file} \
  --tolerance 1e-9 \
  --iter_offset {iter_offset}

echo "Finished at $(date)"
"""


RECOMPUTE_GTAU_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -p super,batch
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH -J recompute_gtau_it_{iteration}
#SBATCH -t 1:00:00
#SBATCH -o recompute_gtau_it_{iteration}.o%j
#SBATCH -e recompute_gtau_it_{iteration}.e%j
#SBATCH --qos special

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Running on $(hostname) at $(date)"

srun -u /home/umyangyu/install/recompute_gtau/bin/recompute_gtau \
  --BETA {beta} \
  --grid_file {ir_file} \
  --input_file {input_file} \
  --results_file {results_file} \
  --iter_offset 0

echo "Finished at $(date)"
"""


IMPURITY_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J imp_{imp}_it_{iteration}             # Job name
#SBATCH -p ludicrous,super
#SBATCH -o imp_{imp}_it_{iteration}.o%j         # Name of stdout output file
#SBATCH -e imp_{imp}_it_{iteration}.e%j         # Name of stderr error file
#SBATCH --exclusive
#SBATCH --qos special
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --hint nomultithread
#SBATCH -t 48:00:00

date
echo $SLURM_NODELIST

export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export LD_LIBRARY_PATH=$HOME/install/arpack/lib64:$LD_LIBRARY_PATH
SolversDir="/home/umyangyu/install/seet_solvers/"

NBATH_PER_ORBITAL={n_bath_per_orbital}             
BETA={beta}                    
NORBITALS={n_orbital_impurity}                  
NBATH_TOTAL=$((NORBITALS * NBATH_PER_ORBITAL))  
NSITES=$((NORBITALS + NBATH_TOTAL)) 
INPUT_FILE="./ed.{imp}.input.h5"
OUTPUT_FILE="./ed.{imp}.result.h5"

srun $SolversDir/bin/ed_solver \
  --arpack.NEV=8 \
  --arpack.NCV=25 \
  --lanc.NOMEGA=1000 \
  --FREQ_FILE={ir_file} \
  --FREQ_PATH=/fermi/ngrid \
  --NSITES=$NSITES \
  --NSPINS={n_spin_impurity} \
  --INPUT_FILE=$INPUT_FILE \
  --OUTPUT_FILE=$OUTPUT_FILE \
  --arpack.SECTOR=false \
  --siam.NORBITALS=$NORBITALS \
  --spinstorage.ORBITAL_NUMBER=$NORBITALS \
  --lanc.BETA=$BETA
  
date
"""


IMPURITY_PREPROCESSING_SCRIPT_TEMPLATE_ESPRIT = """#!/bin/bash
mv ed.{imp}.input.h5 ed.{imp}.input.pre_esprit.h5

hyb_type="from_seet"
hyb_file="./imp_{imp}_delta.txt"
init_dlr_euv=10.0
n_soe_counterterms={n_soe_counterterms}
n_soe_hybridization={n_soe_hybridization}
do_subtraction_trick=true
separate_spins=true
output_dir="./"
hopping_file="$output_dir/imp_{imp}_hopping.txt"
uijkl_file="$output_dir/imp_{imp}_Uijkl_chem.txt"
counterterm_exe="$HOME/src/QMBToolkit/QMBToolkitSOEHYB.jl/scripts/gen_soe_counterterms_hyb_light.jl"
txt_to_h5_exe="$HOME/src/QMBToolkit/QMBToolkitSOEHYB.jl/scripts/seet/convert_txt_to_ed_h5.jl"

mkdir -p $output_dir

julia $counterterm_exe \
    --beta {beta} \
    --n_spin_impurity {n_spin_impurity} \
    --n_orbital_impurity {n_orbital_impurity} \
    --n_tau {n_tau} \
    --hyb_type $hyb_type \
    --hyb_file $hyb_file \
    --init_dlr_euv $init_dlr_euv \
    --hopping_file $hopping_file \
    --n_soe_counterterms $n_soe_counterterms \
    --n_soe_hybridization $n_soe_hybridization \
    --output_dir $output_dir \
    --do_subtraction_trick $do_subtraction_trick \
    --separate_spins $separate_spins

mv hyb_poles.txt imp_{imp}_hyb_poles.txt
mv hyb_weights.txt imp_{imp}_hyb_weights.txt
mv soe_error.txt imp_{imp}_soe_error.txt

julia $txt_to_h5_exe \
    --uijkl_file $uijkl_file \
    --hopping_file $hopping_file \
    --output_h5_file ./ed.{imp}.input.h5 \
    --seet_reference_file "./ed.{imp}.input.pre_esprit.h5" \
    --n_orb {n_orbital_impurity} \
    --n_spin {n_spin_impurity} \
    --mu 0.0 \
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import time
import h5py
import numpy as np
from green_mbtools.pesto.ir import IR_factory

def write_script(path: Path, content: str):
    path.write_text(content)
    path.chmod(0o755)

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
        # Job may have been purged from scontrol; fall back to sacct
        return _check_job_status_sacct(job_id)

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

def _check_job_status_sacct(job_id: str) -> str:
    """
    Fallback status check via sacct for jobs no longer tracked by scontrol.
    """
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "UNKNOWN"
    # sacct may return multiple lines (job + job steps); use the first line
    state = result.stdout.strip().splitlines()[0].strip()
    if state == "COMPLETED":
        return "COMPLETED"
    elif state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
        return "FAILED"
    elif state in ("RUNNING", "PENDING"):
        return state
    else:
        return "UNKNOWN"

def wait_for_jobs(job_ids: list, check_interval: int = 30, max_wait: int = 186400) -> bool:
    """
    Wait for all jobs to complete.
    Polls at a fixed interval of check_interval seconds.
    Returns True if all jobs completed successfully, False otherwise.
    """
    remaining = set(job_ids)
    prev_statuses = {}
    elapsed = 0
    while remaining and elapsed < max_wait:
        for job_id in list(remaining):
            status = check_job_status(job_id)
            if status != prev_statuses.get(job_id):
                print(f"  Job {job_id}: {status}")
                prev_statuses[job_id] = status
            if status == "COMPLETED":
                remaining.discard(job_id)
            elif status in ("FAILED", "UNKNOWN"):
                return False
        if not remaining:
            return True
        time.sleep(check_interval)
        elapsed += check_interval
    if remaining:
        print(f"  Jobs {remaining} timed out after {max_wait}s")
        return False
    return True

def wait_for_job(job_id: str, check_interval: int = 30, max_wait: int = 186400) -> bool:
    """
    Wait for a single job to complete.
    """
    return wait_for_jobs([job_id], check_interval=check_interval, max_wait=max_wait)

def collect_iteration_files(iteration: int, workdir: Path):
    """
    Create a folder named itXXX and move all files matching *itXXX* into it.
    """
    iter_dir = workdir / f"it_{iteration}"
    iter_dir.mkdir(exist_ok=True)
    pattern1 = f"*it_{iteration}.*" 
    pattern2 = f"*it_{iteration}_*" 
    for f in workdir.glob(pattern1):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))
    for f in workdir.glob(pattern2):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))
    for f in workdir.glob(f"imp*.*"):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))
    for f in workdir.glob("dc.*result.h5"):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))
    for f in workdir.glob("ed.*.h5"):
        if f.is_file():
            shutil.move(str(f), str(iter_dir / f.name))
    # remove rm -rf G_ij_omega G_omega
    for f in workdir.glob("G_ij_omega*"):
        if f.is_file():
            f.unlink()
    for f in workdir.glob("G_omega*"):
        if f.is_file():
            f.unlink()

def process_impurity_input(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    """
    Process impurity input files for a given iteration.
    """
    if iteration == 1:
        results_file = args.weak_coupling_dir / "scgw/NiO_GW_1000_lastiter.h5"
    else:
        results_file = args.results_file
    preprocess_script = IMPURITY_PREPROCESSING_SCRIPT_TEMPLATE.format(
        iteration=iteration, beta=args.beta, ir_file=args.ir_file,
        transform_file=args.transform_file, results_file=results_file,
        bath_file=workdir / args.bath_file, input_file=args.input_file,
        iter_offset=1 if iteration > 1 else 0
    )
    preprocess_script_path = workdir / f"preprocess_it_{iteration}.sh"
    write_script(preprocess_script_path, preprocess_script)
    preprocess_job_id = submit_slurm_job(preprocess_script_path)
    if not wait_for_job(preprocess_job_id, check_interval=args.imp_check_interval):
        print(f"Preprocessing job failed for iteration {iteration}.")
        return False
    if args.do_esprit:
        print(f"\n--- Iteration {iteration}/{args.niters}: ESPRIT overwrite ---")
        for imp in range(args.n_impurity):
            preprocess_script = IMPURITY_PREPROCESSING_SCRIPT_TEMPLATE_ESPRIT.format(imp=imp, beta=args.beta, n_tau=args.n_tau, n_spin_impurity=args.n_spin_impurity, n_orbital_impurity=args.n_orbitals_impurity[imp], n_soe_counterterms=args.n_soe_counterterms, n_soe_hybridization=args.n_soe_hybridization)
            preprocess_script_path = workdir / f"preprocess_imp_{imp}_it_{iteration}.sh"
            write_script(preprocess_script_path, preprocess_script)
            result = subprocess.run([str(preprocess_script_path)], capture_output=True, text=True, cwd=workdir)
            if result.returncode != 0:
                print(f"Preprocessing failed for impurity {imp} at iteration {iteration}: {result.stderr}")

    return True

def process_impurity_output(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    """
    Process impurity output files for a given iteration.
    """

    sigma_tau_impurity = []
    sigma_inf_impurity = []
    ir = IR_factory(args.beta, args.ir_file)

    with h5py.File(args.transform_file, 'r') as ftransform:
        X_inv_k_full = ftransform['X_inv_k'][()]
        uu_trans = []
        for i in range(args.n_impurity):
            uu_i = ftransform[f"{i}/UU"][()] + 0j
            uu_trans.append(uu_i)

    with h5py.File(args.input_file, 'r') as finput:
        ir_list = finput['grid/ir_list'][()]
    X_inv_k = X_inv_k_full[ir_list]

    with h5py.File(args.results_file, 'r') as fsimseet:
        sigma_in = fsimseet[f'iter{iteration}/Selfenergy/data'][()]
        sigma_inf_in = fsimseet[f'iter{iteration}/Sigma1'][()]
    ntau_ir, ns, nk, n_orbital_full, _ = sigma_in.shape

    for imp in range(args.n_impurity):
        h5py_file = workdir / f"ed.{imp}.result.h5"
        with h5py.File(h5py_file, 'r') as fimp:
            Sigma_ij_raw = fimp['results/Sigma_ij'][()]
            Sigma_inf_ij = fimp['results/Sigma_inf_ij'][()]

        # ALPSCore stores complex data as interleaved real/imag floats:
        # shape (nfreq, nspin, norb, norb*2) -> (nfreq, nspin, norb, norb) complex
        Sigma_ij = Sigma_ij_raw[..., 0::2] + 1j * Sigma_ij_raw[..., 1::2]
        sigma_tau_IR = ir.w_to_tau(Sigma_ij)
        sigma_tau_impurity.append(sigma_tau_IR)
        sigma_inf_impurity.append(Sigma_inf_ij)
    

    # Active space to full orthogonal basis and combine SIGMA from all impurity blocks
    sigma_tau_local_orth = np.zeros((ntau_ir, ns, n_orbital_full, n_orbital_full), dtype=np.complex128)
    sigma_inf_local_orth = np.zeros((ns, n_orbital_full, n_orbital_full), dtype=np.complex128)
    for i in range(args.n_impurity):
        sigma_tau_local_orth += np.einsum('pi, tspq, qj -> tsij', uu_trans[i].conj(), sigma_tau_impurity[i], uu_trans[i])
        sigma_inf_local_orth += np.einsum('pi, spq, qj -> sij', uu_trans[i].conj(), sigma_inf_impurity[i], uu_trans[i])

    # Orthogonal to AO basis for each k-point: X_inv_k @ sigma_local @ X_inv_k†
    # X_inv_k transforms from orthogonal to AO basis (X_k is AO→orth, its inverse is orth→AO)
    X_inv_k_H = X_inv_k.conj().transpose(0, 2, 1)  # (nk, norb, norb)
    # For iteration 1 SEET skips mixing (sigma_mixing returns early when iter==1),
    # so the full impurity contribution must be added. For iter > 1, SEET applies
    # mixing_weight * sigma_new + (1-mixing_weight) * sigma_old, so we scale the
    # externally-added impurity part by mixing_weight to match.
    weight = 1.0 if iteration == 1 else args.mixing
    sigma_inf_in += (X_inv_k[None] @ sigma_inf_local_orth[:, None] @ X_inv_k_H[None]) * weight
    for t in range(ntau_ir):
        sigma_in[t] += (X_inv_k[None] @ sigma_tau_local_orth[t][:, None] @ X_inv_k_H[None]) * weight

    # Enforce Hermitian symmetry: S -> (S + S†) / 2, matching the C++ make_hermitian() call
    sigma_inf_in = 0.5 * (sigma_inf_in + sigma_inf_in.conj().transpose(0, 1, 3, 2))
    sigma_in = 0.5 * (sigma_in + sigma_in.conj().transpose(0, 1, 2, 4, 3))

    # Save the data back to results file
    with h5py.File(args.results_file, 'r+') as fsimseet:
        fsimseet[f'iter{iteration}/Selfenergy/data'][...] = sigma_in
        fsimseet[f'iter{iteration}/Sigma1'][...] = sigma_inf_in
    
    return True  # Return True to continue workflow in placeholder mode

def sanity_check(args: argparse.Namespace):
    n_orbitals_impurity_file = []
    with h5py.File(args.transform_file, 'r') as ftransform:
        nimp_file = ftransform['nimp'][()]
        for i in range(nimp_file):
            uu_i = ftransform[f"{i}/UU"][()] + 0j
            n_orbitals_impurity_file.append(uu_i.shape[0])
        
    if nimp_file != args.n_impurity:
        raise ValueError(f"Number of impurities in transform file ({nimp_file}) does not match --n_impurity ({args.n_impurity})")
    
    if n_orbitals_impurity_file != args.n_orbitals_impurity:
        raise ValueError(f"Number of orbitals in transform file ({n_orbitals_impurity_file}) does not match --n_orbitals_impurity ({args.n_orbitals_impurity})")

def main():
    parser = argparse.ArgumentParser(description="Submit SEET + SOEHYB solver jobs in sequence")
    parser.add_argument("--niters", type=int, default=2, help="Number of SEET iterations to run")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Working directory for job scripts")
    parser.add_argument("--seet-check-interval", type=int, default=10, help="Initial polling interval (seconds) for SEET job status, doubles each round")
    parser.add_argument("--imp-check-interval", type=int, default=10, help="Initial polling interval (seconds) for impurity job status, doubles each round")
    parser.add_argument("--transform_file", type=Path, default="../transform.h5", help="Path to transformation file")
    parser.add_argument("--results_file", type=Path, default="seet_results.h5", help="Path to SEET results file")
    parser.add_argument("--input_file", type=Path, default="/pauli-storage/gharsha/NiO/dft/input.h5", help="Path to DFT input file (for IBZ k-point mapping)")
    parser.add_argument("--mixing", type=float, default=0.4, help="Mixing parameter for self-energy update")
    parser.add_argument("--ir_file", type=Path, default="/home/umyangyu/src/green-grids/data/ir/1e4.h5", help="Path to IR grid file")
    parser.add_argument("--beta", type=float, default=100.0, help="Inverse temperature for IR grid")
    parser.add_argument("--n_tau", type=int, default=1001, help="Number of tau points for impurity solver")
    parser.add_argument("--n_spin_impurity", type=int, default=2, help="Number of spins in a single impurity model")
    parser.add_argument("--n_orbitals_impurity", type=int, nargs='+', default=[2,2], help="Number of orbitals in each impurity model (list of length n_impurity)")
    parser.add_argument("--weak_coupling_dir", type=Path, default="/pauli-storage/gharsha/NiO/", help="Directory containing weak coupling results (GW)")
    parser.add_argument("--bath_file", type=Path, default="../bath_2bpo.txt", help="Path to bath file generated by impurity preprocessing step")
    parser.add_argument("--n_bath_per_orbital", type=int, default=2, help="Number of bath orbitals per impurity orbital")
    parser.add_argument("--do_esprit", type=bool, default=True, help="Whether to use ESPRIT for bath fitting")
    parser.add_argument("--n_soe_counterterms", type=int, default=1, help="Number of SOE counterterms to generate in preprocessing")
    parser.add_argument("--n_soe_hybridization", type=int, default=14, help="Number of SOE hybridization terms to generate in preprocessing")

    args = parser.parse_args()
    args.n_impurity = len(args.n_orbitals_impurity)
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    sanity_check(args)
    job_ids = []

    ## Temporarily copy reference file for benchmarking: copy sim.seet.h5,
    ## remove the last iteration, and skip all iterations except the last.
    # if args.niters > 1:
        # ref_file = Path('../sim.seet.h5')
        # print(f"Benchmarking: copying {ref_file} -> {args.results_file} and removing last iteration")
        # shutil.copy(str(ref_file), str(args.results_file))
        # with h5py.File(args.results_file, 'r+') as f:
        #     n_iter = int(f['iter'][()])
        #     if f'iter{n_iter}' in f:
        #         del f[f'iter{n_iter}']
        #     f['iter'][...] = n_iter - 1

    for iteration in range(1, args.niters+1):
        # if iteration != args.niters:
        #     continue

        try:
            # ---- SEET ----
            restart = "false" if iteration == 1 else "true"
            seet_script_path = workdir / f"seet_it_{iteration}.sh"
            seet_script = SEET_SCRIPT_TEMPLATE.format(
                iteration=iteration, beta=args.beta, transform_filename=args.transform_file,
                results_filename=args.results_file, mixing=args.mixing, ir_file=args.ir_file, restart=restart, weak_coupling_dir=args.weak_coupling_dir
            )
            write_script(seet_script_path, seet_script)

            print(f"\n--- Iteration {iteration}/{args.niters}: SEET ---")
            seet_job_id = submit_slurm_job(seet_script_path)
            job_ids.append(("SEET", iteration, seet_job_id))
            if not wait_for_job(seet_job_id, check_interval=args.seet_check_interval):
                print(f"SEET job failed at iteration {iteration}. Stopping.")
                break

            # ---- Preprocessing ----
            print(f"\n--- Iteration {iteration}/{args.niters}: Preprocessing ---")
            if not process_impurity_input(iteration, workdir, args):
                print(f"Preprocessing failed at iteration {iteration}. Stopping.")
                break

            # ---- Impurity solvers ----
            print(f"\n--- Iteration {iteration}/{args.niters}: Impurity solvers ---")
            impurity_job_ids = []
            for imp in range(args.n_impurity):
                imp_script_path = workdir / f"imp_{imp}_it_{iteration}.sh"
                imp_script = IMPURITY_SCRIPT_TEMPLATE.format(
                    iteration=iteration, imp=imp, beta=args.beta, n_bath_per_orbital=args.n_bath_per_orbital, n_orbital_impurity=args.n_orbitals_impurity[imp], ir_file=args.ir_file, n_spin_impurity=args.n_spin_impurity
                )
                write_script(imp_script_path, imp_script)
                impurity_job_id = submit_slurm_job(imp_script_path)
                impurity_job_ids.append(impurity_job_id)
                job_ids.append(("Impurity", iteration, impurity_job_id))

            if not wait_for_jobs(impurity_job_ids, check_interval=args.imp_check_interval):
                print(f"Impurity solver failed at iteration {iteration}. Stopping.")
                break
            
            print(f"\n--- Iteration {iteration}/{args.niters}: Processing impurity output ---")
            if not process_impurity_output(iteration, workdir, args):
                print(f"Failed to process impurity output at iteration {iteration}. Stopping.")
                break

            # ---- Recompute G(tau) and energies from corrected sigma ----
            print(f"\n--- Iteration {iteration}/{args.niters}: Recompute G(tau) and energies ---")
            recompute_script = RECOMPUTE_GTAU_SCRIPT_TEMPLATE.format(
                iteration=iteration, beta=args.beta, ir_file=args.ir_file,
                input_file=args.input_file, results_file=args.results_file
            )
            recompute_script_path = workdir / f"recompute_gtau_it_{iteration}.sh"
            write_script(recompute_script_path, recompute_script)
            recompute_job_id = submit_slurm_job(recompute_script_path)
            job_ids.append(("RecomputeG", iteration, recompute_job_id))
            if not wait_for_job(recompute_job_id, check_interval=args.imp_check_interval):
                print(f"Recompute G(tau) failed at iteration {iteration}. Stopping.")
                break
        finally:
            collect_iteration_files(iteration, workdir)
        

    
    print(f"\n--- Summary ---")
    for job_type, it, job_id in job_ids:
        print(f"  {job_type:10} it_{it}: {job_id}")


if __name__ == "__main__":
    main()
