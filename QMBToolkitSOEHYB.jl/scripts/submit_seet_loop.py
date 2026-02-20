#!/usr/bin/env python3
SEET_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J seet_it_{iteration}
#SBATCH -p ludicrous,super,batch
#SBATCH -o seet_it_{iteration}.o%j
#SBATCH -e seet_it_{iteration}.e%j
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH -t 48:00:00

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
  --weak_results $WeakCouplingDir/scgw/NiO_GW.h5 \
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
  --dc_data_prefix "dc_int" \
  --seet_root_dir "./" \
  --spin_symm false \
  --verbose 4

date
"""


IMPURITY_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J imp_{imp}_it_{iteration}             # Job name
#SBATCH -p ludicrous,super,batch
#SBATCH -o imp_{imp}_it_{iteration}.o%j         # Name of stdout output file
#SBATCH -e imp_{imp}_it_{iteration}.e%j         # Name of stderr error file
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -c 1
#SBATCH --hint nomultithread
#SBATCH -t 48:00:00

date
echo $SLURM_NODELIST

export OPENBLAS_NUM_THREADS=1
PROJECT_DIR=~/src/QMBToolkit/QMBToolkitSOEHYB.jl

# Precompile the package once before launching MPI tasks
# This prevents all MPI ranks from trying to compile simultaneously
julia --check-bounds=no --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); using QMBToolkitSOEHYB'

srun julia --check-bounds=no --project=$PROJECT_DIR -e '
using QMBToolkitSOEHYB
QMBToolkitSOEHYB.run("{input_file}")
'

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
save_linear_size = {n_tau}

[solver.dlr]
rtol = 1e-12
Euv = 20.0

[solver]
output_path = "./imp_{imp}_sim.h5"
"""

IMPURITY_PREPROCESSING_SCRIPT_TEMPLATE = """#!/bin/bash
hyb_type="from_seet"
hyb_file="./imp_{imp}_delta.txt"
init_dlr_euv=10.0
n_soe_counterterms=1
n_soe_hybridization=14
do_subtraction_trick=true
separate_spins=true
output_dir="./"
hopping_file="$output_dir/imp_{imp}_hopping.txt"
counterterm_exe="$HOME/src/QMBToolkit/QMBToolkitSOEHYB.jl/scripts/gen_soe_counterterms_hyb_light.jl"

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
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import time
import h5py
from green_mbtools.pesto.ir import IR_factory
import numpy as np
from scipy import interpolate

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

def wait_for_jobs(job_ids: list, check_interval: int = 30, max_wait: int = 86400) -> bool:
    """
    Wait for all jobs to complete.
    Uses exponential backoff: starts polling at check_interval seconds,
    doubling each round up to a maximum of 64 * check_interval.
    Returns True if all jobs completed successfully, False otherwise.
    """
    remaining = set(job_ids)
    prev_statuses = {}
    elapsed = 0
    current_interval = check_interval
    max_interval = check_interval * 64
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
        time.sleep(current_interval)
        elapsed += current_interval
        current_interval = min(current_interval * 2, max_interval)
    if remaining:
        print(f"  Jobs {remaining} timed out after {max_wait}s")
        return False
    return True

def wait_for_job(job_id: str, check_interval: int = 30, max_wait: int = 86400) -> bool:
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


def create_impurity_input(iteration: int, imp: int, workdir: Path, args: argparse.Namespace):
    """
    Create input file for impurity solver based on template.
    """
    input_file = workdir / f"imp_{imp}_it_{iteration}.toml"
    input_content = IMPURITY_INPUT_TEMPLATE.format(
        beta=args.beta,
        imp=imp,
        expansion_order_propagator=args.expansion_order,
        expansion_order_green=args.expansion_order+1,
        n_tau=args.n_tau
    )
    input_file.write_text(input_content)

def process_impurity_input(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    """
    Process impurity input files for a given iteration.
    """
    for imp in range(args.n_impurity):
        preprocess_script = IMPURITY_PREPROCESSING_SCRIPT_TEMPLATE.format(imp=imp, beta=args.beta, n_tau=args.n_tau, n_spin_impurity=args.n_spin_impurity, n_orbital_impurity=args.n_orbitals_impurity[imp])
        preprocess_script_path = workdir / f"preprocess_imp_{imp}_it_{iteration}.sh"
        write_script(preprocess_script_path, preprocess_script)
        result = subprocess.run([str(preprocess_script_path)], capture_output=True, text=True, cwd=workdir)
        if result.returncode != 0:
            print(f"Preprocessing failed for impurity {imp} at iteration {iteration}: {result.stderr}")
            return False
        else:
            print(f"Preprocessing completed for impurity {imp} at iteration {iteration}")
    return True

def _read_delta_tau_from_txt(filename, beta):
    data = np.loadtxt(filename)

    # All indices in file are 0-based
    tau_indices = data[:, 0].astype(int)
    i_indices = data[:, 1].astype(int)
    j_indices = data[:, 2].astype(int)
    real_parts = data[:, 3]

    n_tau = np.max(tau_indices) + 1
    n_flav = max(np.max(i_indices), np.max(j_indices)) + 1

    # Convert tau indices to actual tau values: [0, beta]
    tau_grid = np.linspace(0, beta, n_tau)

    delta_tau = np.zeros((n_tau, n_flav, n_flav), dtype=np.complex128)

    delta_tau[tau_indices, i_indices, j_indices] = real_parts

    return delta_tau, tau_grid

def _from_flavor_to_spin_orbital(f, n_spin=2):
    n, n_flavor, _ = f.shape
    n_orb = n_flavor // n_spin

    f_so = np.zeros((n, n_spin, n_orb, n_orb), dtype=np.complex128)

    for spin in range(n_spin):
        for orb1 in range(n_orb):
            for orb2 in range(n_orb):
                # Converted to 0-based indexing
                flavor1 = n_spin * orb1 + spin
                flavor2 = n_spin * orb2 + spin
                f_so[:, spin, orb1, orb2] = f[:, flavor1, flavor2]

    return f_so

def read_delta_tau_from_txt(delta_file: str, n_orbital: int, beta: float):
    delta_tau, tau_grid = _read_delta_tau_from_txt(delta_file, beta)
    delta_tau_so = _from_flavor_to_spin_orbital(delta_tau)
    return delta_tau_so, tau_grid

def read_hopping_from_txt(hopping_file: str, n_orbital: int, n_spin=2):
    n_flav = n_spin * n_orbital
    entries = []
    with open(hopping_file) as k:
        for raw_line in k:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            i_str, m_str, ij_str_re, ij_str_im = line.split()
            entries.append((int(i_str), int(m_str), float(ij_str_re), float(ij_str_im)))
    max_idx = max(max(i, m) for i, m, _, _ in entries) + 1
    hopping_raw_full = np.zeros((max_idx, max_idx), dtype=complex)
    for i, m, re, im in entries:
        hopping_raw_full[i, m] = complex(re + 1j * im)
    # Extract impurity block only (first n_flav rows/cols), discard bath orbitals
    hopping_raw = hopping_raw_full[:n_flav, :n_flav]
    hopping = _from_flavor_to_spin_orbital(hopping_raw[np.newaxis, :, :], n_spin=n_spin)[0]
    return hopping

def interpolation(tau_grid_in, X_tau, tau_grid_new, kind="cubic"):
    tau_grid_in = np.asarray(tau_grid_in, dtype=float)
    ntau_in = len(tau_grid_in)
    tau_grid_new = np.asarray(tau_grid_new, dtype=float)
    ntau_out = len(tau_grid_new)
    X_tau = np.asarray(X_tau)

    # merge indices except tau and interpolate
    orig_shape = X_tau.shape
    X_tau = X_tau.reshape((ntau_in, -1))

    # initialize output array
    new_delta_tau = np.zeros((ntau_out, X_tau.shape[-1]), dtype=complex)

    for m in range(X_tau.shape[1]):
        real_interp = interpolate.interp1d(
            tau_grid_in, X_tau[:, m].real,
            kind=kind, fill_value="extrapolate", assume_sorted=True
        )
        imag_interp = interpolate.interp1d(
            tau_grid_in, X_tau[:, m].imag,
            kind=kind, fill_value="extrapolate", assume_sorted=True
        )
        new_delta_tau[:, m] = real_interp(tau_grid_new) + 1j * imag_interp(tau_grid_new)
    
    # reshape to original shape with new tau dimension
    new_delta_tau = new_delta_tau.reshape((ntau_out,) + orig_shape[1:])

    return new_delta_tau

def process_impurity_output(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    """
    Process impurity output files for a given iteration.
    """

    sigma_tau_impurity = []
    sigma_inf_impurity = []
    ir = IR_factory(args.beta, args.ir_file)
    tau_grid_ir = ir.tau_mesh

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
        group = fsimseet[f'iter{iteration}/Selfenergy']
        mu = fsimseet[f'iter{iteration}/mu'][()]
        sigma_in = fsimseet[f'iter{iteration}/Selfenergy/data'][()]
        sigma_inf_in = fsimseet[f'iter{iteration}/Sigma1'][()]
    ntau_ir, ns, nk, n_orbital_full, _ = sigma_in.shape

    for imp, n_orbital_impurity in enumerate(args.n_orbitals_impurity):
        h5py_file = workdir / f"imp_{imp}_sim.h5"
        with h5py.File(h5py_file, 'r') as fimp:
            tau_grid_green = fimp['green/linear_grid/tau'][()]
            green_tau_f = fimp['green/linear_grid/green_tau'][()]
        green_tau = _from_flavor_to_spin_orbital(green_tau_f, n_spin=args.n_spin_impurity)
        delta_tau, tau_grid_delta = read_delta_tau_from_txt(str(workdir / f"imp_{imp}_delta.txt"), n_orbital_impurity, args.beta)
        hopping = read_hopping_from_txt(str(workdir / f"imp_{imp}_hopping.txt"), n_orbital_impurity)

        delta_tau_IR = interpolation(tau_grid_delta, delta_tau, tau_grid_ir, kind="cubic")
        green_tau_IR = interpolation(tau_grid_green, green_tau, tau_grid_ir, kind="cubic")

        # Fourier transform to Matsubara frequency
        delta_omega_IR = ir.tau_to_w(delta_tau_IR)
        green_omega_IR = ir.tau_to_w(green_tau_IR)

        # Compute self-energy in Matsubara frequency
        # G0_inv(iw, s) = (iw + mu) * I - hopping[s] - delta(iw, s)
        I = np.eye(n_orbital_impurity, dtype=complex)
        g0_inv = (1j * ir.wsample[:, None, None, None] + mu) * I - hopping[None] - delta_omega_IR
        g_inv = np.linalg.inv(green_omega_IR)
        sigma_omega_IR = g0_inv - g_inv

        # Approximate Sigma_inf as sigma(iw_max) — the value at the highest
        # Matsubara frequency on the IR grid. This is subtracted per spin and
        # orbital channel so that sigma_omega_IR decays to zero at high
        # frequency, enabling a clean Fourier transform back to tau.
        sigma_inf = sigma_omega_IR[-1].copy()
        sigma_omega_IR -= sigma_inf[None]

        # Fourier transform
        sigma_tau_IR = ir.w_to_tau(sigma_omega_IR)
        
        sigma_tau_impurity.append(sigma_tau_IR)
        sigma_inf_impurity.append(sigma_inf)
    

    # Active space to full orthogonal basis and combine SIGMA from all impurity blocks
    sigma_tau_local_orth = np.zeros((ntau_ir, ns, n_orbital_full, n_orbital_full), dtype=np.complex128)
    sigma_inf_local_orth = np.zeros((ns, n_orbital_full, n_orbital_full), dtype=np.complex128)
    for i in range(args.n_impurity):
        sigma_tau_local_orth += np.einsum('pi, tspq, qj -> tsij', uu_trans[i].conj(), sigma_tau_impurity[i], uu_trans[i])
        sigma_inf_local_orth += np.einsum('pi, spq, qj -> sij', uu_trans[i].conj(), sigma_inf_impurity[i], uu_trans[i])

    # Orthogonal to AO basis for each k-point: X_inv_k @ sigma_local @ X_inv_k†
    # X_inv_k transforms from orthogonal to AO basis (X_k is AO→orth, its inverse is orth→AO)
    X_inv_k_H = X_inv_k.conj().transpose(0, 2, 1)  # (nk, norb, norb)
    weight = 1.0 if iteration == 1 else args.mixing
    sigma_inf_in += (X_inv_k[None] @ sigma_inf_local_orth[:, None] @ X_inv_k_H[None]) * weight
    for t in range(ntau_ir):
        sigma_in[t] += (X_inv_k[None] @ sigma_tau_local_orth[t][:, None] @ X_inv_k_H[None]) * weight

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
    parser.add_argument("--niters", type=int, default=10, help="Number of SEET iterations to run")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Working directory for job scripts")
    parser.add_argument("--seet-check-interval", type=int, default=30, help="Initial polling interval (seconds) for SEET job status, doubles each round")
    parser.add_argument("--imp-check-interval", type=int, default=30, help="Initial polling interval (seconds) for impurity job status, doubles each round")
    parser.add_argument("--transform_file", type=Path, default="transform.h5", help="Path to transformation file")
    parser.add_argument("--results_file", type=Path, default="seet_results.h5", help="Path to SEET results file")
    parser.add_argument("--input_file", type=Path, default="/pauli-storage/gharsha/NiO/dft/input.h5", help="Path to DFT input file (for IBZ k-point mapping)")
    parser.add_argument("--mixing", type=float, default=0.4, help="Mixing parameter for self-energy update")
    parser.add_argument("--ir_file", type=Path, default="/home/umyangyu/src/green-grids/data/ir/1e4.h5", help="Path to IR grid file")
    parser.add_argument("--beta", type=float, default=100.0, help="Inverse temperature for IR grid")
    parser.add_argument("--expansion_order", type=int, default=1, help="Expansion order for impurity solver")
    parser.add_argument("--n_tau", type=int, default=1001, help="Number of tau points for impurity solver")
    parser.add_argument("--n_spin_impurity", type=int, default=2, help="Number of spins in a single impurity model")
    parser.add_argument("--n_orbitals_impurity", type=int, nargs='+', default=[2,2], help="Number of orbitals in each impurity model (list of length n_impurity)")
    parser.add_argument("--weak_coupling_dir", type=Path, default="/pauli-storage/gharsha/NiO", help="Directory containing weak coupling results (GW)")

    args = parser.parse_args()
    args.n_impurity = len(args.n_orbitals_impurity)
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    sanity_check(args)
    job_ids = []
    
    for iteration in range(1, args.niters+1):
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
                create_impurity_input(iteration, imp, workdir, args)
                imp_script_path = workdir / f"imp_{imp}_it_{iteration}.sh"
                imp_script = IMPURITY_SCRIPT_TEMPLATE.format(
                    iteration=iteration, imp=imp, input_file=f"imp_{imp}_it_{iteration}.toml"
                )
                write_script(imp_script_path, imp_script)
                impurity_job_id = submit_slurm_job(imp_script_path)
                impurity_job_ids.append(impurity_job_id)
                job_ids.append(("Impurity", iteration, impurity_job_id))

            if not wait_for_jobs(impurity_job_ids, check_interval=args.imp_check_interval):
                print(f"Impurity solver failed at iteration {iteration}. Stopping.")
                break

            if not process_impurity_output(iteration, workdir, args):
                print(f"Failed to process impurity output at iteration {iteration}. Stopping.")
                break
        finally:
            collect_iteration_files(iteration, workdir)
    
    print(f"\n--- Summary ---")
    for job_type, it, job_id in job_ids:
        print(f"  {job_type:10} it_{it}: {job_id}")


if __name__ == "__main__":
    main()
