#!/usr/bin/env python3
"""
Script to submit SEET + inchworm solver jobs in sequence.
Alternates between SEET (with itermax=1) and inchworm solver jobs.

NOTE:
- This is a template script. You need to fill in the actual SEET and inchworm solver commands.
- The processing of inchworm output is a placeholder and needs to be implemented based on your requirements.
- This script will only support SIGMA_MIXING mode for now; CDIIS will require further implementation.
"""

import numpy as np
import h5py
import argparse
import subprocess
import time
from pathlib import Path
from green_mbtools.pesto.ir import IR_factory
from scipy import interpolate
import scipy.linalg as LA


def read_hopping_from_txt(hopping_file: str, nao: int, uhf: bool = True):
    """_summary_

    Parameters
    ----------
    hopping_file : str
        Path to hopping file
    nao : int
        Number of atomic orbitals
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    np.ndarray
        Hopping matrix in UHF format if uhf is True, else in RHF/GHF format
    """
    if uhf:
        number_of_orbitals = 2 * nao
    hopping_raw = np.zeros((number_of_orbitals, number_of_orbitals), dtype=complex)
    with open(hopping_file) as k:
        for raw_line in k:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            i_str, m_str, ij_str_re, ij_str_im = line.split()
            i = int(i_str)
            m = int(m_str)
            hopping_raw[i, m] = complex(float(ij_str_re) + 1j * float(ij_str_im))
    if uhf:
        hopping_uhf = np.zeros((2, nao, nao), dtype=complex)
        hopping_uhf[0, :, :] = hopping_raw[0::2, 0::2]
        hopping_uhf[1, :, :] = hopping_raw[1::2, 1::2]
    else:
        hopping_uhf = hopping_raw.reshape((1, ) + hopping_raw.shape)
    return hopping_uhf


def read_delta_tau_from_txt(delta_file: str, nao: int, beta: float, uhf: bool = True):
    """Read hybridization function from text file.

    Parameters
    ----------
    delta_file : str
        Path to delta file
    nao : int
        Number of atomic orbitals
    beta : float
        Inverse temperature
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # ---------- pass 1: determine max l = even tau grid index ----------
    l_max = -1
    with open(delta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue  # or raise, if you prefer strict
            l = int(parts[0])
            if l > l_max:
                l_max = l

    if l_max < 0:
        raise ValueError(f"No data lines found in {delta_file}")
    n_tau = l_max + 1

    # ---------- allocate ----------
    if uhf:
        ns = 2
    else:
        ns = 1
    delta_tau = np.zeros((n_tau, ns, nao, nao), dtype=complex)

    # ---------- pass 2: fill ----------
    with open(delta_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            l_str, i_str, m_str, re_str, im_str = line.split()
            l = int(l_str)
            i = int(i_str)
            m = int(m_str)
            s1_idx = i % ns
            ao_i = i // ns
            s2_idx = m % ns
            ao_m = m // ns
            if s1_idx != s2_idx:
                continue  # skip spin-flip terms in UHF mode
            delta_tau[l, s1_idx, ao_i, ao_m] = float(re_str) + 1j * float(im_str)

    tau_delta = np.linspace(0.0, beta, n_tau, endpoint=True)

    return delta_tau, tau_delta


def read_greenfunction_from_txt(nao: int, time_filename: str, green_path: str, uhf=True):
    """Read green's function from text files.

    Parameters
    ----------
    nao : int
        Number of atomic orbitals
    time_filename : str
        Path to file containing time points
    green_path : str
        Directory containing green's function files
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the green's function array and the time array
    """
    if uhf:
        number_of_orbitals = 2 * nao
        ns = 2
    else:
        number_of_orbitals = nao
        ns = 1

    # ---------- read time points ----------
    t_list = []
    with open(time_filename) as k:
        for line in k:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # t_str, ij_str = line.split()
            t_str = line.split()[0]
            t_str = t_str.strip().strip('[],')   # remove [ ] and commas
            if t_str:                            # skip empty tokens
                t_list.append(float(t_str))
    t_arr = np.array(t_list)    
    t_shape = len(t_arr)
    
    # ---------- allocate green's function array ----------
    green_tau = np.zeros((t_shape, ns, nao, nao), dtype=complex)
    for i in range(number_of_orbitals):
        for j in range(number_of_orbitals):
            # data = np.loadtxt(f'/home/orit/VS_codes1/example/G_{i}_{j}.dat')
            i_ao = i // ns
            j_ao = j // ns
            s_i = i % ns
            s_j = j % ns
            if s_i != s_j:
                continue # skip spin-flip terms in UHF mode
            with open(f'{green_path}/G_{i}_{j}.dat') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    t_str, ij_str = line.split()            # split into 2 parts
                    # TODO: Waiting for confirmation
                    green_tau[:, s_i, i_ao, j_ao] = np.array([complex(x) for x in ij_str.strip().split(',')])
    return green_tau, t_arr


def interpolation(tau_grid_in, X_tau, tau_grid_new, kind="cubic"):
    """Interpolate X(tau) from tau_grid_in to tau_grid_new
    Parameters
    ----------
    tau_grid_in : np.ndarray
        Original tau grid
    X_tau : np.ndarray
        Data to be interpolated, with tau as the first dimension
    tau_grid_new : np.ndarray
        New tau grid
    kind : str, optional
        Interpolation kind, by default "cubic"
    
    Returns
    -------
    np.ndarray
        Interpolated data on new tau grid
    """
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

def get_inchworm_selfenergy(green_path, time_file, hopping_file, delta_file, nao_imp, ir_file, beta, mu, uhf=True):
    """Extract static and dynamic self-energies on IR grid from, e.g., Inchworm data

    Parameters
    ----------
    green_path : str
        Path to directory containing green's function files
    time_file : str
        Path to file containing time points
    hopping_file : str
        Path to hopping file
    delta_file : str
        Path to delta file
    nao_imp : int
        Number of impurity orbitals
    ir_file : str
        Path to IR file
    beta : float
        Inverse temperature
    mu : float
        Chemical potential
    uhf : bool, optional
        Whether the data is for unrestricted Hartree-Fock, by default True

    Returns
    -------
    _type_
        _description_
    """

    # Set up IR grid
    myir = IR_factory(beta, ir_file)
    tau_grid_ir = myir.tau_mesh

    green_tau, tau_grid_green = read_greenfunction_from_txt(nao_imp, time_file, green_path, uhf=uhf)
    delta_tau, tau_grid_delta = read_delta_tau_from_txt(delta_file, nao_imp, beta, uhf=uhf)
    hopping = read_hopping_from_txt(hopping_file, nao_imp, uhf=uhf)

    # Interpolate to IR grid
    delta_tau_IR = interpolation(tau_grid_delta, delta_tau, tau_grid_ir, kind="cubic")
    green_tau_IR = interpolation(tau_grid_green, green_tau, tau_grid_ir, kind="cubic")

    # Fourier transform to Matsubara frequency
    delta_omega_IR = myir.tau_to_w(delta_tau_IR)
    green_omega_IR = myir.tau_to_w(green_tau_IR)

    # Compute self-energy in Matsubara frequency
    ns = 2 if uhf else 1
    sigma_omega_IR = np.zeros_like(green_omega_IR, dtype=complex)
    for s in range(ns):
        for iw, w_val in enumerate(myir.wsample):
            g0_inv = (1j * w_val + mu) * np.eye(nao_imp, dtype=complex) - hopping[s] - delta_omega_IR[iw, s]
            g_inv = LA.inv(green_omega_IR[iw, s])
            sigma_omega_IR[iw, s] = g0_inv - g_inv

    # Extract Sigma_inf = sigma (iw_max)
    sigma_inf = sigma_omega_IR[-1]
    for iw, w_val in enumerate(myir.wsample):
        sigma_omega_IR[iw] -= sigma_inf

    # Fourier transform
    sigma_tau_IR = myir.w_to_tau(sigma_omega_IR)

    return sigma_inf, sigma_tau_IR


SEET_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=seet_iter_{iter}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=seet_{iter}_%j.log

# Load modules, set up environment, etc.
# module load ...

# Run SEET with inchworm solver and itermax=1
seet_command_here --input input.h5 --itermax 1 --restart true \
  --mixing_type SIGMA_MIXING --mixing_weight {mixing} \
  --results_file {results_filename} --seet_input {transform_filename} \
  --impurity_solver "INCHWORM" ...
"""

INCHWORM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=inchworm_iter_{iter}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=inchworm_{iter}_%j.log

# Load modules, set up environment, etc.
# module load ...

# Run inchworm solver
inchworm_solver_command_here
"""


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


def process_inchworm_output(iteration: int, workdir: Path, args: argparse.Namespace) -> bool:
    """
    Process inchworm solver output before next SEET iteration.
    
    PLACEHOLDER: Implement your post-processing logic here.
    
    Examples of what you might do:
    - Read inchworm output files
    - Extract self-energy, Green's function, or other observables
    - Transform/convert data formats
    - Update input files for next SEET iteration
    - Check convergence criteria
    - Copy/move files to appropriate locations
    
    Parameters
    ----------
    iteration : int
        Current iteration number (starts from 1)
    workdir : Path
        Working directory containing input/output files
    args : argparse.Namespace
        Command-line arguments containing results_file and transform_file
    
    Returns
    -------
    bool
        True if processing succeeded, False if it failed
    """
    print(f"\n{'='*60}")
    print(f"Processing inchworm output from iteration {iteration}")
    print(f"{'='*60}")

    # Open and obtain transformation
    ftransform = h5py.File(args.transform_file, 'r')
    nimp = ftransform['nimp'][()]
    # AO <-> Orthogonal basis transformations
    X_k = ftransform['X_k'][()]
    # Projection from active to full orbital space
    uu_trans = []
    nao_imp = []
    for i in range(nimp):
        uu_i = ftransform[f"{i}/UU"][()] + 0j
        uu_trans.append(uu_i)
        nao_imp.append(uu_i.shape[0])
    ftransform.close()
    # NOTE: expecting sigma on imaginary-time grid.

    # Open output file
    fsimseet = h5py.File(args.results_file, 'r+')
    group = fsimseet['iter{}/Selfenergy'.format(iteration)]
    mu = group['mu'][()]
    sigma_in = group['data'][()]
    sigma_inf_in = fsimseet['iter{}/Sigma1'.format(iteration)][()]
    ntau, ns, nk, nao_full, _ = sigma_in.shape
    
    # PLACEHOLDER: 
    # 1. Read inchworm output files
    # 2. Extract new local self-energy
    #       NOTE: Separate static and dynamic contributions not needed, but highly appreciated
    # 3. Store them in the variable sigma_new of shape (ntau, nao_imp, nao_imp) where
    #       ntau is the number of imaginary time points on IR grid
    #       nao_imp is the number of impurity orbitals

    ntau = 100  # Placeholder -- remove if needed
    ns_imp = 2  # Placeholder number of spin -- remove if needed
    assert ns_imp == ns, "Spin dimension mismatch"
    sigma_tau_inchworm = []
    sigma_inf_inchworm = []
    default_green_func_path = '.'
    default_time_filename = 'time_intervals.txt'
    for i in range(nimp):
        # NOTE: Replace `get_inchworm_selfenergy` with a custom defined function for other applications
        sigma_inf, sigma_tau = get_inchworm_selfenergy(
            default_green_func_path, default_time_filename, f'imp_{i}_hopping.txt', f'imp_{i}_delta.txt',
            nao_imp[i], args.ir_file, args.BETA, mu, args.uhf
        )
        sigma_inf_inchworm.append(sigma_inf)
        sigma_tau_inchworm.append(sigma_tau)

    # Active space to full orthogonal basis and combine SIGMA from all impurity blocks
    sigma_tau_local_orth = np.zeros((ntau, ns, nao_full, nao_full), dtype=np.complex128)
    sigma_inf_local_orth = np.zeros((ns, nao_full, nao_full), dtype=np.complex128)
    for i in range(nimp):
        sigma_tau_local_orth += np.einsum('pi, tspq, qj -> tsij', uu_trans[i].conj(), sigma_tau_inchworm[i], uu_trans[i])
        sigma_inf_local_orth += np.einsum('pi, spq, qj -> sij', uu_trans[i].conj(), sigma_inf_inchworm[i], uu_trans[i])

    # Orthogonal to AO basisi for each k-point
    sigma_loc_ao_ts_ibz = np.zeros((ntau, ns, nk, nao_full, nao_full), dtype=np.complex128)
    sigma_loc_ao_inf_ibz = np.zeros((ns, nk, nao_full, nao_full), dtype=np.complex128)
    for t in range(ntau):
        for s in range(ns):
            sigma_loc_ao_ts_ibz[t, s] = np.einsum('kab, bc, sdc -> kad', X_k, sigma_tau_local_orth[t, s], X_k.conj())
            sigma_loc_ao_inf_ibz[s] = np.einsum('kab, bc, sdc -> kad', X_k, sigma_inf_local_orth[s], X_k.conj())
    
    # Update SIGMA from results file
    for t in range(ntau):
        for s in range(ns):
            sigma_in[t, s] += sigma_loc_ao_ts_ibz[t, s] * args.mixing
            sigma_inf_in[s] += sigma_loc_ao_inf_ibz[s] * args.mixing
    
    # Save the data back to results file
    group['data'][...] = sigma_in
    fsimseet['iter{}/Sigma1'.format(iteration)][...] = sigma_inf_in
    fsimseet.close()
    
    print("[PLACEHOLDER] Inchworm output processing not implemented")
    print("              Add your processing logic to process_inchworm_output()")
    return True  # Return True to continue workflow in placeholder mode


def main():
    parser = argparse.ArgumentParser(description="Submit SEET + inchworm solver jobs in sequence")
    parser.add_argument("--niters", type=int, default=3, help="Number of SEET iterations to run")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Working directory for job scripts")
    parser.add_argument("--check-interval", type=int, default=30, help="Interval (seconds) to check job status")
    parser.add_argument("--dry-run", action="store_true", help="Print job scripts without submitting")
    parser.add_argument("--transform_file", type=Path, default="transform.h5", help="Path to transformation file")
    parser.add_argument("--results_file", type=Path, default="seet_results.h5", help="Path to SEET results file")
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter for self-energy update")
    parser.add_argument("--ir_file", type=Path, default="1e5.h5", help="Path to IR grid file")
    parser.add_argument("--BETA", type=float, default=100.0, help="Inverse temperature for IR grid")
    parser.add_argument("--uhf", type=int, default=1, help="Urestricted calculations or not.")
    
    args = parser.parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    
    job_ids = []
    
    for iteration in range(1, args.niters + 1):
        # ===== SEET job =====
        seet_script_path = workdir / f"seet_iter_{iteration}.sh"
        seet_script = SEET_SCRIPT_TEMPLATE.format(
            iter=iteration, transform_filename=args.transform_file,
            results_filename=args.results_file, mixing=args.mixing
        )
        seet_script_path.write_text(seet_script)
        seet_script_path.chmod(0o755)
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Submitting SEET job")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"[DRY-RUN] Would submit: {seet_script_path}")
            print(seet_script)
        else:
            seet_job_id = submit_slurm_job(seet_script_path)
            job_ids.append(("SEET", iteration, seet_job_id))
            
            # Wait for SEET to complete
            if not wait_for_job(seet_job_id, check_interval=args.check_interval):
                print(f"SEET job failed at iteration {iteration}. Stopping.")
                break
        
        # ===== Inchworm job =====
        inchworm_script_path = workdir / f"inchworm_iter_{iteration}.sh"
        inchworm_script = INCHWORM_SCRIPT_TEMPLATE.format(iter=iteration)
        inchworm_script_path.write_text(inchworm_script)
        inchworm_script_path.chmod(0o755)
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Submitting inchworm solver job")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"[DRY-RUN] Would submit: {inchworm_script_path}")
            print(inchworm_script)
        else:
            inchworm_job_id = submit_slurm_job(inchworm_script_path)
            job_ids.append(("Inchworm", iteration, inchworm_job_id))
            
            # Wait for inchworm to complete
            if not wait_for_job(inchworm_job_id, check_interval=args.check_interval):
                print(f"Inchworm job failed at iteration {iteration}. Stopping.")
                break
            
            # Process inchworm output before next SEET iteration
            if not process_inchworm_output(iteration, workdir, args):
                print(f"Failed to process inchworm output at iteration {iteration}. Stopping.")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("Job submission summary:")
    print(f"{'='*60}")
    for job_type, iteration, job_id in job_ids:
        print(f"{job_type:12} Iteration {iteration}: Job ID {job_id}")


if __name__ == "__main__":
    main()
