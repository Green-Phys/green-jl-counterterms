#!/usr/bin/env julia
#
# Manual demonstration that checkpoint/restart works under MPI and produces correct
# results after a real mid-run kill — at both order granularity and within-order
# (group-chunked) granularity.
#
# Run from the package root:
#
#     julia --project=. test/checkpoint_restart_mpi.jl
#
# It launches the solver as separate `mpiexec` jobs (so a "crash" is a genuine
# process exit, not an in-process throw) and checks:
#
#   Order-level (checkpoint_group_interval = 0):
#   1. Reference run at -np 2 completes and deletes its checkpoint.
#   2. Kill at -np 2, then resume at -np 2  => bit-identical to the reference
#      (same rank count => deterministic group ranges + per-block Allreduce).
#   3. Kill at -np 2, then resume at -np 3  => equal to ~1e-12 relative but NOT
#      bitwise (floating-point summation is non-associative across rank counts).
#      This is correct, not a bug — do not "fix" it by forcing bit-equality.
#
#   Within-order (checkpoint_group_interval > 0; killed MID-order via the group hook):
#   4. Reference run at -np 2. NOTE this is NOT bit-identical to run 1: chunking
#      regroups the floating-point sums, so within-order runs are compared against
#      this same-interval reference, not against run 1.
#   5. Mid-order kill at -np 2, then resume at -np 2  => bit-identical to run 4.
#   6. Mid-order kill at -np 2, then resume at -np 3  => ~1e-12 relative (different
#      rank count), same rationale as run 3.
#
# The kill is triggered by the built-in test hooks: QMBTOOLKIT_CRASH_PHASE /
# QMBTOOLKIT_CRASH_AFTER_ORDER (order boundary) or additionally
# QMBTOOLKIT_CRASH_AFTER_GROUPS (within order). The rank-0 checkpoint is written,
# all ranks barrier, then every rank errors — leaving a complete checkpoint on disk
# exactly as an out-of-walltime kill would.

using MPI
using HDF5
using Printf

const PKG = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG, "test", "data")
const RUNNER = "using QMBToolkitSOEHYB; QMBToolkitSOEHYB.run(ARGS[1])"

function write_input(dir; interval = 0)
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
        checkpoint_group_interval = $(interval)

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

# Launch the solver as an mpiexec job. Pass at most one of:
#   `crash = (phase, order)`              — kill at an order boundary, or
#   `group_crash = (phase, order, groups)` — kill mid-order after `groups` groups.
# Returns true on success (exit 0), false otherwise.
#
# `setenv(cmd, base.env)` re-applies the launcher environment that MPI.jl wraps
# around the mpiexec binary (library paths etc.) — backtick interpolation of a
# Cmd does not carry its env, so without this the subprocess fails to start.
function launch(np::Int, input::String; crash = nothing, group_crash = nothing)
    base = MPI.mpiexec()
    cmd = `$base -np $np $(Base.julia_cmd()) --project=$PKG -e $RUNNER $input`
    cmd = setenv(cmd, base.env)
    if crash !== nothing
        cmd = addenv(cmd, "QMBTOOLKIT_CRASH_PHASE" => crash[1],
            "QMBTOOLKIT_CRASH_AFTER_ORDER" => string(crash[2]))
    elseif group_crash !== nothing
        cmd = addenv(cmd, "QMBTOOLKIT_CRASH_PHASE" => group_crash[1],
            "QMBTOOLKIT_CRASH_AFTER_ORDER" => string(group_crash[2]),
            "QMBTOOLKIT_CRASH_AFTER_GROUPS" => string(group_crash[3]))
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

# Green order 3 has 32 diagram groups, so an interval of 4 gives within-order
# checkpoints after groups 4, 8, ..., 28. We kill after 8 (a chunk boundary).
const GROUP_INTERVAL = 4
const GROUP_CRASH = ("green", 3, 8)

# --- 4. Within-order reference run at -np 2 ----------------------------------
refw_dir = mktempdir()
refw_in = write_input(refw_dir; interval = GROUP_INTERVAL)
println("[4] within-order reference run at -np 2 (checkpoint_group_interval=$GROUP_INTERVAL) ...")
@assert launch(2, refw_in) "within-order reference run failed"
@assert !isfile(joinpath(refw_dir, "sim.ckpt.h5")) "within-order reference did not delete its checkpoint"
refw = read_arrays(joinpath(refw_dir, "sim.h5"))
println("    done; checkpoint deleted on success ✓")

# --- 5. Mid-order kill at -np 2, resume at -np 2 (expect bit-identical) ------
d5 = mktempdir()
in5 = write_input(d5; interval = GROUP_INTERVAL)
println("[5] mid-order kill at -np 2 (green order 3 after 8 groups), then resume at -np 2 ...")
@assert !launch(2, in5; group_crash = GROUP_CRASH) "within-order crash run unexpectedly succeeded"
@assert isfile(joinpath(d5, "sim.ckpt.h5")) "no checkpoint after mid-order kill"
@assert launch(2, in5) "within-order resume at -np 2 failed"
@assert !isfile(joinpath(d5, "sim.ckpt.h5")) "checkpoint not deleted after within-order resume"
got5 = read_arrays(joinpath(d5, "sim.h5"))
@assert bitidentical(got5, refw) "within-order resume at same nprocs must be bit-identical"
println("    within-order resume at -np 2 is BIT-IDENTICAL to the within-order reference ✓")

# --- 6. Mid-order kill at -np 2, resume at -np 3 (expect ~1e-12) -------------
d6 = mktempdir()
in6 = write_input(d6; interval = GROUP_INTERVAL)
println("[6] mid-order kill at -np 2 (green order 3 after 8 groups), then resume at -np 3 ...")
@assert !launch(2, in6; group_crash = GROUP_CRASH) "within-order crash run unexpectedly succeeded"
@assert isfile(joinpath(d6, "sim.ckpt.h5")) "no checkpoint after mid-order kill"
@assert launch(3, in6) "within-order resume at -np 3 failed"
@assert !isfile(joinpath(d6, "sim.ckpt.h5")) "checkpoint not deleted after within-order resume"
got6 = read_arrays(joinpath(d6, "sim.h5"))
@assert close_rtol(got6, refw, 1e-12) "within-order resume at different nprocs must match to ~1e-12"
@printf("    within-order resume at -np 3 matches the within-order reference to rtol<=1e-12 (bitwise-equal: %s) ✓\n",
    bitidentical(got6, refw))

println("\nALL MPI CHECKPOINT/RESTART CHECKS PASSED")
