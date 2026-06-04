# Checkpoint / restart verification.
#
# A crash is simulated with the test-only hook (QMBTOOLKIT_CRASH_PHASE /
# QMBTOOLKIT_CRASH_AFTER_ORDER) which barriers then errors AFTER the order's
# checkpoint is on disk — modelling a realistic mid-run kill. We then rerun the
# identical command (auto-resume) and require the resumed sim.h5 to be EXACTLY
# equal to an uninterrupted reference run.
#
# Exact `==` (not isapprox) is intentional: at a single rank the per-rank group
# range and per-block Allreduce are deterministic, and HDF5 round-trips Float64
# losslessly, so a resumed run is bit-identical. (Across different nprocs the
# result is still correct but only equal to ~1e-16 relative; see the MPI example.)

function _read_sim_arrays(path::String)
    h5open(path, "r") do f
        (
            Z = read(f, "/partition_function/partition_function_by_order"),
            g1 = read(f, "/green/by_order/green_order_1"),
            g2 = read(f, "/green/by_order/green_order_2"),
            g3 = read(f, "/green/by_order/green_order_3"),
            g_dlr = read(f, "/green/dlr_grid/green_tau"),
            g_coeffs = read(f, "/green/dlr_grid/coeffs"),
            prop_info = read(f, "/diagram_info/propagator_by_order"),
            green_info = read(f, "/diagram_info/green_by_order")
        )
    end
end

function _assert_identical(got, ref)
    @test got.Z == ref.Z
    @test got.g1 == ref.g1
    @test got.g2 == ref.g2
    @test got.g3 == ref.g3
    @test got.g_dlr == ref.g_dlr
    @test got.g_coeffs == ref.g_coeffs
    @test got.prop_info == ref.prop_info
    @test got.green_info == ref.green_info
end

# Crash at (phase, order), then auto-resume; assert the checkpoint lifecycle and
# bit-identical result against `ref`.
function _crash_then_resume(phase::String, order::Int, ref)
    test_dir = mktempdir()
    input_file = joinpath(test_dir, "input.toml")
    write(input_file, generate_input_toml())
    ckpt = joinpath(test_dir, "sim.ckpt.h5")

    threw = false
    ENV["QMBTOOLKIT_CRASH_PHASE"] = phase
    ENV["QMBTOOLKIT_CRASH_AFTER_ORDER"] = string(order)
    try
        QMBToolkitSOEHYB.run(input_file)
    catch
        threw = true
    finally
        delete!(ENV, "QMBTOOLKIT_CRASH_PHASE")
        delete!(ENV, "QMBTOOLKIT_CRASH_AFTER_ORDER")
    end
    @test threw                 # the crash hook fired
    @test isfile(ckpt)          # checkpoint persisted before the crash

    QMBToolkitSOEHYB.run(input_file)   # auto-resume
    @test isfile(joinpath(test_dir, "sim.h5"))
    @test !isfile(ckpt)         # deleted on successful completion

    _assert_identical(_read_sim_arrays(joinpath(test_dir, "sim.h5")), ref)
end

function run_checkpoint_restart_test()
    # Reference: uninterrupted run.
    ref_dir = mktempdir()
    ref_input = joinpath(ref_dir, "input.toml")
    write(ref_input, generate_input_toml())
    QMBToolkitSOEHYB.run(ref_input)
    @test isfile(joinpath(ref_dir, "sim.h5"))
    @test !isfile(joinpath(ref_dir, "sim.ckpt.h5"))   # deleted on success
    ref = _read_sim_arrays(joinpath(ref_dir, "sim.h5"))

    # Crash points spanning both phases, including the propagator->green
    # boundary (propagator order == P, propagator fully done before any green).
    @testset "Crash at propagator order $k" for k in 0:2
        _crash_then_resume("propagator", k, ref)
    end
    @testset "Crash at green order $k" for k in 1:2
        _crash_then_resume("green", k, ref)
    end

    # Escape hatch: with a checkpoint present, QMBTOOLKIT_NO_RESUME=1 ignores it
    # and runs fresh, still producing the identical result.
    @testset "QMBTOOLKIT_NO_RESUME ignores checkpoint" begin
        test_dir = mktempdir()
        input_file = joinpath(test_dir, "input.toml")
        write(input_file, generate_input_toml())
        ckpt = joinpath(test_dir, "sim.ckpt.h5")

        ENV["QMBTOOLKIT_CRASH_PHASE"] = "green"
        ENV["QMBTOOLKIT_CRASH_AFTER_ORDER"] = "1"
        try
            QMBToolkitSOEHYB.run(input_file)
        catch
        finally
            delete!(ENV, "QMBTOOLKIT_CRASH_PHASE")
            delete!(ENV, "QMBTOOLKIT_CRASH_AFTER_ORDER")
        end
        @test isfile(ckpt)

        ENV["QMBTOOLKIT_NO_RESUME"] = "1"
        try
            QMBToolkitSOEHYB.run(input_file)
        finally
            delete!(ENV, "QMBTOOLKIT_NO_RESUME")
        end
        @test !isfile(ckpt)
        _assert_identical(_read_sim_arrays(joinpath(test_dir, "sim.h5")), ref)
    end
end

# --- Within-order (sub-order) checkpoint / restart --------------------------
#
# With `checkpoint_group_interval = C > 0`, each order's diagram groups are
# processed in consecutive prefix chunks of C groups; after each chunk the partial
# sum is Allreduced into the cumulative accumulator and a checkpoint records that
# accumulator plus the integer count of groups done. A crash mid-order is simulated
# with the group hook (QMBTOOLKIT_CRASH_AFTER_GROUPS); resume continues from the
# next chunk and must be bit-identical to an UNINTERRUPTED run at the SAME interval
# and nprocs (same chunk boundaries => same per-chunk FP operations replayed).
#
# Note: a given C>0 differs from C=0 only at ~1e-16 relative (FP summation is
# regrouped across chunk boundaries), so the science is unchanged but the bytes are
# not bit-identical to C=0 — hence each interval is compared against its OWN
# uninterrupted reference for the exact `==` checks, and against the C=0 reference
# only with rtol=1e-12.

# Uninterrupted run at a given within-order checkpoint interval; returns sim arrays.
function _run_with_interval(interval::Int)
    d = mktempdir()
    inp = joinpath(d, "input.toml")
    write(inp, generate_input_toml(; checkpoint_group_interval = interval))
    QMBToolkitSOEHYB.run(inp)
    @test isfile(joinpath(d, "sim.h5"))
    @test !isfile(joinpath(d, "sim.ckpt.h5"))   # deleted on success
    return _read_sim_arrays(joinpath(d, "sim.h5"))
end

# Crash mid-order via the group hook (after `groups` groups of `order` in `phase`),
# verify the checkpoint metadata, then auto-resume; assert bit-identical to `ref`
# (an uninterrupted run at the same interval and nprocs). `groups` must be a chunk
# boundary strictly below the order's valid-group count.
function _crash_then_resume_within(phase::String, order::Int, groups::Int, interval::Int, ref)
    d = mktempdir()
    inp = joinpath(d, "input.toml")
    write(inp, generate_input_toml(; checkpoint_group_interval = interval))
    ckpt = joinpath(d, "sim.ckpt.h5")

    threw = false
    ENV["QMBTOOLKIT_CRASH_PHASE"] = phase
    ENV["QMBTOOLKIT_CRASH_AFTER_ORDER"] = string(order)
    ENV["QMBTOOLKIT_CRASH_AFTER_GROUPS"] = string(groups)
    try
        QMBToolkitSOEHYB.run(inp)
    catch
        threw = true
    finally
        delete!(ENV, "QMBTOOLKIT_CRASH_PHASE")
        delete!(ENV, "QMBTOOLKIT_CRASH_AFTER_ORDER")
        delete!(ENV, "QMBTOOLKIT_CRASH_AFTER_GROUPS")
    end
    @test threw                 # the within-order crash hook fired
    @test isfile(ckpt)          # checkpoint persisted before the crash

    # The mid-order checkpoint records the in-flight group count and the highest
    # FULLY completed order (= order-1) in the right phase.
    h5open(ckpt, "r") do f
        @test read(f["meta/phase"]) == phase
        @test Int(read(f["meta/in_flight_groups"])) == groups
        @test Int(read(f["meta/last_completed_order"])) == order - 1
    end

    QMBToolkitSOEHYB.run(inp)   # auto-resume from the next chunk
    @test isfile(joinpath(d, "sim.h5"))
    @test !isfile(ckpt)         # deleted on successful completion

    _assert_identical(_read_sim_arrays(joinpath(d, "sim.h5")), ref)
end

function run_within_order_checkpoint_test()
    ref0 = _run_with_interval(0)
    ref1 = _run_with_interval(1)   # checkpoint after every single group
    ref3 = _run_with_interval(3)   # chunks of 3 groups (uneven final chunk)

    # Chunking must not change the science: any interval reproduces interval=0 to ~1e-12.
    @testset "interval=$C matches interval=0 (rtol 1e-12)" for (C, refC) in ((1, ref1), (3, ref3))
        @test isapprox(refC.Z, ref0.Z; rtol = 1e-12)
        @test isapprox(refC.g1, ref0.g1; rtol = 1e-12)
        @test isapprox(refC.g2, ref0.g2; rtol = 1e-12)
        @test isapprox(refC.g3, ref0.g3; rtol = 1e-12)
    end

    # Within-order resume == uninterrupted run at the same interval, bit-identical.
    # Valid-group counts for the test data (probed): propagator order 2 = 8,
    # green order 2 = 5, green order 3 = 32. Crash points are chunk boundaries
    # strictly below those counts. The propagator scenario also exercises the
    # phase guard: after the propagator resume completes, the green phase must run
    # fresh (group offset must NOT leak across phases).
    @testset "within-order resume interval=1: $phase order $order after $groups groups" for (phase, order, groups) in (("propagator", 2, 3), ("green", 2, 2), ("green", 3, 5))
        _crash_then_resume_within(phase, order, groups, 1, ref1)
    end
    @testset "within-order resume interval=3: green order 3 after 6 groups" begin
        _crash_then_resume_within("green", 3, 6, 3, ref3)
    end
end
