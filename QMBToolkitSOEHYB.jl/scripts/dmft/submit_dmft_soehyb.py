#!/usr/bin/env python3

import argparse
from pathlib import Path
import shutil
import subprocess
import time

# === Templates ===

SOLVER_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J soehyb_it_{iteration}
#SBATCH -p {partition}
#SBATCH -D {iter_dir}
#SBATCH -o soehyb_it_{iteration}.o%j
#SBATCH -e soehyb_it_{iteration}.e%j
#SBATCH --exclusive
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH -c 1
#SBATCH --hint nomultithread
#SBATCH -t {time}

date
echo $SLURM_NODELIST

export OPENBLAS_NUM_THREADS=1
PROJECT_DIR=~/src/QMBToolkit/QMBToolkitSOEHYB.jl

# Precompile once before launching MPI tasks
julia --check-bounds=no --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); using QMBToolkitSOEHYB'

srun julia --check-bounds=no --project=$PROJECT_DIR -e '
using QMBToolkitSOEHYB
QMBToolkitSOEHYB.run("input.toml")
'

date
"""

SOLVER_INPUT_TEMPLATE = """[model]
beta = {beta}
file_path = "./"
hopping_file = "hopping.txt"
interaction_file = "Uijkl.txt"
hybridization_poles_file = "hyb_poles.txt"
hybridization_weights_file = "hyb_weights.txt"

[solver.propagator]
is_bold = false
expansion_order = {expansion_order}

[solver.green]
is_bold = false
expansion_order = {expansion_order_green}
save_linear_size = {n_tau}

[solver.dlr]
rtol = {dlr_rtol}
Euv = {dlr_euv}

[solver]
output_path = "./sim.h5"
"""

# === Script paths ===
SCRIPTS_DIR = Path.home() / "src/QMBToolkit/QMBToolkitSOEHYB.jl/scripts"
DMFT_SCRIPTS_DIR = SCRIPTS_DIR / "dmft"


# === SLURM utilities ===

def write_script(path: Path, content: str):
    path.write_text(content)
    path.chmod(0o755)


def submit_slurm_job(script_path: Path) -> str:
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")
    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted job {job_id}: {script_path.name}")
    return job_id


def check_job_status(job_id: str) -> str:
    result = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True)
    if result.returncode != 0:
        return _check_job_status_sacct(job_id)
    if "JobState=COMPLETED" in result.stdout:
        return "COMPLETED"
    elif "JobState=RUNNING" in result.stdout:
        return "RUNNING"
    elif "JobState=PENDING" in result.stdout:
        return "PENDING"
    elif "JobState=FAILED" in result.stdout or "JobState=TIMEOUT" in result.stdout:
        return "FAILED"
    return "UNKNOWN"


def _check_job_status_sacct(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "UNKNOWN"
    state = result.stdout.strip().splitlines()[0].strip()
    if state == "COMPLETED":
        return "COMPLETED"
    elif state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
        return "FAILED"
    elif state in ("RUNNING", "PENDING"):
        return state
    return "UNKNOWN"


def wait_for_jobs(job_ids: list, check_interval: int = 30, max_wait: int = 9999999) -> bool:
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


def wait_for_job(job_id: str, check_interval: int = 30, max_wait: int = 9999999) -> bool:
    return wait_for_jobs([job_id], check_interval=check_interval, max_wait=max_wait)


# === Julia script runner ===

def run_julia_script(script_path, args_list, description=""):
    cmd = ["julia", str(script_path)] + [str(a) for a in args_list]
    print(f"  Running: {description or script_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT:\n{result.stdout}")
        print(f"  STDERR:\n{result.stderr}")
        raise RuntimeError(f"{script_path.name} failed")
    if result.stdout.strip():
        print(result.stdout)


# === DMFT workflow steps ===

def generate_initial_inputs(workdir: Path, args):
    """Run gen_interaction_hopping.jl to create hopping.txt and Uijkl.txt in workdir."""
    print("\n--- Generating hopping and interaction files ---")
    run_julia_script(
        SCRIPTS_DIR / "gen_interaction_hopping.jl",
        [
            "--n_spin_impurity", args.n_spin,
            "--n_orbital_impurity", args.n_orbital,
            "--t", args.t_hopping,
            "--U", args.U,
            "--J", args.J,
            "--Uprime", args.Uprime,
            "--mu", args.mu,
            "--output_dir", workdir,
        ],
        "gen_interaction_hopping.jl"
    )


def run_write_hyb_G0(iteration: int, workdir: Path, iter_dir: Path, args):
    """Run write_hyb_G0.jl to generate delta.txt and G0(iw)."""
    print(f"\n--- Iteration {iteration}: write_hyb_G0 ---")
    script_args = [
        "--beta", args.beta,
        "--t", args.t,
        "--n_spin", args.n_spin,
        "--n_orbital", args.n_orbital,
        "--n_tau", args.n_tau,
        "--iteration", iteration,
        "--output_file", iter_dir / "delta.txt",
        "--hopping_file", workdir / "hopping.txt",
        "--dlr_euv", args.dlr_euv,
        "--dlr_rtol", args.dlr_rtol,
        "--dmft_h5", workdir / "dmft.h5",
        "--off_diagonal_prefactor", args.off_diagonal_prefactor,
    ]
    if iteration > 1:
        prev_iter_dir = workdir / f"it_{iteration - 1}"
        script_args += ["--prev_sim_h5", prev_iter_dir / "sim.h5"]
    if args.zero_off_diagonal:
        script_args += ["--zero_off_diagonal", "true"]

    run_julia_script(
        DMFT_SCRIPTS_DIR / "write_hyb_G0.jl",
        script_args,
        "write_hyb_G0.jl"
    )


def run_preprocessing(iteration: int, iter_dir: Path, args):
    """Run gen_soe_counterterms_hyb_light.jl for SOE fitting and counterterms."""
    print(f"\n--- Iteration {iteration}: SOE preprocessing ---")
    run_julia_script(
        SCRIPTS_DIR / "gen_soe_counterterms_hyb_light.jl",
        [
            "--beta", args.beta,
            "--n_spin_impurity", args.n_spin,
            "--n_orbital_impurity", args.n_orbital,
            "--n_tau", args.n_tau,
            "--hyb_type", "from_seet",
            "--hyb_file", iter_dir / "delta.txt",
            "--init_dlr_euv", args.dlr_euv,
            "--hopping_file", iter_dir / "hopping.txt",
            "--n_soe_counterterms", args.n_soe_counterterms,
            "--n_soe_hybridization", args.n_soe_hybridization,
            "--output_dir", iter_dir,
            "--do_subtraction_trick", "true" if args.do_subtraction_trick else "false",
            "--separate_spins", "false",
        ],
        "gen_soe_counterterms_hyb_light.jl"
    )


def create_solver_input(iteration: int, iter_dir: Path, args):
    """Create TOML input file for the solver."""
    input_content = SOLVER_INPUT_TEMPLATE.format(
        beta=args.beta,
        expansion_order=args.expansion_order,
        expansion_order_green=args.expansion_order + 1,
        n_tau=args.n_tau,
        dlr_rtol=args.dlr_rtol,
        dlr_euv=args.dlr_euv,
    )
    (iter_dir / "input.toml").write_text(input_content)


def run_compute_sigma(iteration: int, iter_dir: Path, workdir: Path, args):
    """Run compute_sigma.jl to compute self-energy from G0 and G."""
    print(f"\n--- Iteration {iteration}: compute_sigma ---")
    run_julia_script(
        DMFT_SCRIPTS_DIR / "compute_sigma.jl",
        [
            "--iteration", iteration,
            "--sim_h5", iter_dir / "sim.h5",
            "--dmft_h5", workdir / "dmft.h5",
            "--beta", args.beta,
            "--dlr_euv", args.dlr_euv,
            "--dlr_rtol", args.dlr_rtol,
            "--n_matsubara", args.n_matsubara,
        ],
        "compute_sigma.jl"
    )


# === Main ===

def main():
    parser = argparse.ArgumentParser(
        description="DMFT loop for Bethe lattice with SOE-HYB solver"
    )

    # Physical parameters
    parser.add_argument("--beta", type=float, required=True, help="Inverse temperature")
    parser.add_argument("--U", type=float, required=True, help="Coulomb interaction")
    parser.add_argument("--J", type=float, default=0.0, help="Hund's coupling")
    parser.add_argument("--Uprime", type=float, default=None,
                        help="Inter-orbital Coulomb (default: U - 2J)")
    parser.add_argument("--mu", type=float, required=True, help="Chemical potential")
    parser.add_argument("--t_hopping", type=float, default=0.0,
                        help="Inter-orbital hopping in impurity Hamiltonian")
    parser.add_argument("--t", type=float, default=1.0,
                        help="Bethe lattice hopping for self-consistency (delta = t^2 * G)")

    # Model
    parser.add_argument("--n_spin", type=int, default=2, help="Number of spin flavors")
    parser.add_argument("--n_orbital", type=int, required=True, help="Number of orbitals")

    # Grid / DLR (consistent across all scripts)
    parser.add_argument("--n_tau", type=int, default=2001, help="Linear tau grid size")
    parser.add_argument("--dlr_euv", type=float, default=20.0, help="DLR energy cutoff Euv")
    parser.add_argument("--dlr_rtol", type=float, default=1e-14, help="DLR relative tolerance")

    # Solver
    parser.add_argument("--expansion_order", type=int, default=1,
                        help="Propagator expansion order (Green order = this + 1)")

    # SOE
    parser.add_argument("--n_soe_counterterms", type=int, default=1,
                        help="Number of SOE counterterms")
    parser.add_argument("--n_soe_hybridization", type=int, default=14,
                        help="Number of SOE hybridization terms")
    parser.add_argument("--no-subtraction-trick", dest="do_subtraction_trick",
                        action="store_false", help="Disable subtraction trick for SOE")
    parser.set_defaults(do_subtraction_trick=True)

    # DMFT
    parser.add_argument("--niters", type=int, default=10, help="Number of DMFT iterations")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Working directory")
    parser.add_argument("--off_diagonal_prefactor", type=float, default=0.0,
                        help="Off-diagonal prefactor for initial semicircular hybridization")
    parser.add_argument("--zero_off_diagonal", action="store_true",
                        help="Zero off-diagonal hybridization and delta(iw) before G0 computation")

    # SLURM
    parser.add_argument("--check_interval", type=int, default=60,
                        help="SLURM polling interval in seconds")
    parser.add_argument("--partition", type=str, default="ludicrous",
                        help="SLURM partition")
    parser.add_argument("--nodes", type=int, default=1, help="Number of SLURM nodes")
    parser.add_argument("--ntasks", type=int, default=64, help="Number of MPI tasks")
    parser.add_argument("--time", type=str, default="48:00:00", help="SLURM wall time limit")

    args = parser.parse_args()

    # Compute defaults
    if args.Uprime is None:
        args.Uprime = args.U - 2.0 * args.J

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    print("=== DMFT Loop: Bethe Lattice + SOE-HYB ===")
    print(f"Beta: {args.beta}, U: {args.U}, J: {args.J}, U': {args.Uprime}, mu: {args.mu}")
    print(f"t (Bethe): {args.t}, t_hopping (impurity): {args.t_hopping}")
    print(f"n_spin: {args.n_spin}, n_orbital: {args.n_orbital}")
    print(f"n_tau: {args.n_tau}, DLR Euv: {args.dlr_euv}, DLR rtol: {args.dlr_rtol}")
    print(f"Expansion order: {args.expansion_order} (propagator), "
          f"{args.expansion_order + 1} (Green)")
    print(f"SOE counterterms: {args.n_soe_counterterms}, "
          f"hybridization: {args.n_soe_hybridization}, "
          f"subtraction trick: {args.do_subtraction_trick}")
    print(f"Iterations: {args.niters}, workdir: {workdir}")

    # Phase 1: Generate hopping.txt and Uijkl.txt (master copies in workdir)
    generate_initial_inputs(workdir, args)

    job_ids = []

    # Phase 2: DMFT loop
    for iteration in range(1, args.niters + 1):
        print(f"\n{'=' * 60}")
        print(f"=== DMFT Iteration {iteration}/{args.niters} ===")
        print(f"{'=' * 60}")

        iter_dir = workdir / f"it_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        # Copy fresh hopping.txt and Uijkl.txt (counterterms will modify hopping)
        shutil.copy(workdir / "hopping.txt", iter_dir / "hopping.txt")
        shutil.copy(workdir / "Uijkl.txt", iter_dir / "Uijkl.txt")

        # Write delta.txt and G0(iw) to dmft.h5
        run_write_hyb_G0(iteration, workdir, iter_dir, args)

        # SOE preprocessing: fits SOE, adds counterterms to hopping, writes hyb_poles/weights
        run_preprocessing(iteration, iter_dir, args)

        # Create solver input TOML
        create_solver_input(iteration, iter_dir, args)

        # Submit solver SLURM job
        print(f"\n--- Iteration {iteration}: Submitting solver ---")
        solver_script = SOLVER_SCRIPT_TEMPLATE.format(
            iteration=iteration,
            partition=args.partition,
            nodes=args.nodes,
            ntasks=args.ntasks,
            time=args.time,
            iter_dir=iter_dir,
        )
        solver_script_path = iter_dir / f"soehyb_it_{iteration}.sh"
        write_script(solver_script_path, solver_script)
        job_id = submit_slurm_job(solver_script_path)
        job_ids.append((iteration, job_id))

        if not wait_for_job(job_id, check_interval=args.check_interval):
            print(f"Solver failed at iteration {iteration}. Stopping.")
            break

        # Compute self-energy: Sigma = G0^{-1} - G^{-1}, save G and Sigma to dmft.h5
        run_compute_sigma(iteration, iter_dir, workdir, args)

    # Summary
    print(f"\n{'=' * 60}")
    print("--- Summary ---")
    for it, job_id in job_ids:
        status = check_job_status(job_id)
        print(f"  Iteration {it}: job {job_id} ({status})")
    print(f"DMFT data: {workdir / 'dmft.h5'}")


if __name__ == "__main__":
    main()
