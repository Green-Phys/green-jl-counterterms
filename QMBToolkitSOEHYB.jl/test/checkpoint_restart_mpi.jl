#!/usr/bin/env julia
#
# Manual demonstration that order-level checkpoint/restart works under MPI and
# produces correct results after a real mid-run kill.
#
# Run from the package root:
#
#     julia --project=. test/checkpoint_restart_mpi.jl
#
# It launches the solver as separate `mpiexec` jobs (so a "crash" is a genuine
# process exit, not an in-process throw) and checks three things:
#
#   1. Reference run at -np 2 completes and deletes its checkpoint.
#   2. Kill at -np 2, then resume at -np 2  => bit-identical to the reference
#      (same rank count => deterministic group ranges + per-block Allreduce).
#   3. Kill at -np 2, then resume at -np 3  => equal to ~1e-12 relative but NOT
#      bitwise (floating-point summation is non-associative across rank counts).
#      This is correct, not a bug — do not "fix" it by forcing bit-equality.
#
# The kill is triggered by the built-in test hook
# (QMBTOOLKIT_CRASH_PHASE / QMBTOOLKIT_CRASH_AFTER_ORDER): the rank-0 checkpoint
# is written, all ranks barrier, then every rank errors — leaving a complete
# checkpoint on disk exactly as an out-of-walltime kill would.

using MPI
using HDF5
using Printf

const PKG = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG, "test", "data")
const RUNNER = "using QMBToolkitSOEHYB; QMBToolkitSOEHYB.run(ARGS[1])"

function write_input(dir)
    open(joinpath(dir, "input.toml"), "w") do io
        write(io, """
        [model]
        beta = 10.0
        file_path = "$(DATA_DIR)"
        hopping_file = "hopping.txt"
        interaction_file = "Uijkl.txt"
        hybridization_poles_file = "hyb_poles.txt"
        hybridization_weights_file = "hyb_weights.txt"

        [solver]
        output_path = "$(dir)"

        [solver.propagator]
        is_bold = false
        expansion_order = 2

        [solver.green]
        is_bold = false
        expansion_order = 3
        save_linear_size = 1001

        [solver.dlr]
        Euv = 500.0
        rtol = 1e-14
        """)
    end
    return joinpath(dir, "input.toml")
end

# Launch the solver as an mpiexec job. `crash` is `nothing` or `(phase, order)`.
# Returns true on success (exit 0), false otherwise.
#
# `setenv(cmd, base.env)` re-applies the launcher environment that MPI.jl wraps
# around the mpiexec binary (library paths etc.) — backtick interpolation of a
# Cmd does not carry its env, so without this the subprocess fails to start.
function launch(np::Int, input::String; crash = nothing)
    base = MPI.mpiexec()
    cmd = `$base -np $np $(Base.julia_cmd()) --project=$PKG -e $RUNNER $input`
    cmd = setenv(cmd, base.env)
    if crash !== nothing
        cmd = addenv(cmd, "QMBTOOLKIT_CRASH_PHASE" => crash[1],
            "QMBTOOLKIT_CRASH_AFTER_ORDER" => string(crash[2]))
    end
    try
        run(pipeline(cmd; stdout = devnull, stderr = devnull))
        return true
    catch
        return false
    end
end

function read_arrays(path)
    h5open(path, "r") do f
        (
            Z = read(f, "/partition_function/partition_function_by_order"),
            g1 = read(f, "/green/by_order/green_order_1"),
            g2 = read(f, "/green/by_order/green_order_2"),
            g3 = read(f, "/green/by_order/green_order_3")
        )
    end
end

bitidentical(a, b) = a.Z == b.Z && a.g1 == b.g1 && a.g2 == b.g2 && a.g3 == b.g3
close_rtol(a, b, rtol) = isapprox(a.Z, b.Z; rtol) && isapprox(a.g1, b.g1; rtol) &&
                         isapprox(a.g2, b.g2; rtol) && isapprox(a.g3, b.g3; rtol)

# --- 1. Reference run at -np 2 ----------------------------------------------
ref_dir = mktempdir()
ref_in = write_input(ref_dir)
println("[1] reference run at -np 2 ...")
@assert launch(2, ref_in) "reference run failed"
@assert !isfile(joinpath(ref_dir, "sim.ckpt.h5")) "reference did not delete its checkpoint"
ref = read_arrays(joinpath(ref_dir, "sim.h5"))
println("    done; checkpoint deleted on success ✓")

# --- 2. Kill at -np 2, resume at -np 2 (expect bit-identical) ----------------
d2 = mktempdir()
in2 = write_input(d2)
println("[2] kill at -np 2 (green order 1), then resume at -np 2 ...")
@assert !launch(2, in2; crash = ("green", 1)) "crash run unexpectedly succeeded"
@assert isfile(joinpath(d2, "sim.ckpt.h5")) "no checkpoint after kill"
@assert launch(2, in2) "resume at -np 2 failed"
@assert !isfile(joinpath(d2, "sim.ckpt.h5")) "checkpoint not deleted after resume"
got2 = read_arrays(joinpath(d2, "sim.h5"))
@assert bitidentical(got2, ref) "resume at same nprocs must be bit-identical"
println("    resume at -np 2 is BIT-IDENTICAL to the reference ✓")

# --- 3. Kill at -np 2, resume at -np 3 (expect correct but not bitwise) ------
d3 = mktempdir()
in3 = write_input(d3)
println("[3] kill at -np 2 (green order 1), then resume at -np 3 ...")
@assert !launch(2, in3; crash = ("green", 1)) "crash run unexpectedly succeeded"
@assert isfile(joinpath(d3, "sim.ckpt.h5")) "no checkpoint after kill"
@assert launch(3, in3) "resume at -np 3 failed"
@assert !isfile(joinpath(d3, "sim.ckpt.h5")) "checkpoint not deleted after resume"
got3 = read_arrays(joinpath(d3, "sim.h5"))
@assert close_rtol(got3, ref, 1e-12) "resume at different nprocs must match to ~1e-12"
@printf("    resume at -np 3 matches the reference to rtol<=1e-12 (bitwise-equal: %s) ✓\n",
    bitidentical(got3, ref))

println("\nALL MPI CHECKPOINT/RESTART CHECKS PASSED")
