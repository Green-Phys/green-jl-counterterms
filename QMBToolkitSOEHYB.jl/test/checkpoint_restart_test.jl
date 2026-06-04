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
