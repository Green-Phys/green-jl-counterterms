# benchmark_end_to_end.jl - Full solver wall time, memory, and GC tracking.
#
# Measures initialization, buffer allocation, propagator/Green evaluation per order,
# total GC time and allocation counts. Includes a warmup pass to eliminate JIT.
#
# Run with:
#     julia --project=. test/benchmark/benchmark_end_to_end.jl

using QMBToolkitSOEHYB
using MPI
using Printf
using LinearAlgebra: diag

const SOE = QMBToolkitSOEHYB

MPI.Init()
comm = MPI.COMM_WORLD

const BENCHMARK_DIR = @__DIR__
const DATA_DIR = joinpath(BENCHMARK_DIR, "data")

# ============================================================================
# Timing infrastructure
# ============================================================================

struct TimingResult
    name::String
    time::Float64       # seconds
    bytes::Int64        # bytes allocated
    gctime::Float64     # GC time in seconds
end

const timings = TimingResult[]

function record!(name, stats)
    push!(timings, TimingResult(name, stats[2], stats[3], stats[4]))
end

function print_timing_table()
    println()
    println("=" ^ 95)
    @printf("  %-45s  %10s  %12s  %10s\n", "Phase", "Time (s)", "Allocs (MiB)", "GC (s)")
    println("-" ^ 95)
    total_t = 0.0
    total_b = Int64(0)
    total_gc = 0.0
    for t in timings
        @printf("  %-45s  %10.3f  %12.1f  %10.3f\n",
            t.name, t.time, t.bytes / (1024^2), t.gctime)
        total_t += t.time
        total_b += t.bytes
        total_gc += t.gctime
    end
    println("-" ^ 95)
    @printf("  %-45s  %10.3f  %12.1f  %10.3f\n",
        "TOTAL", total_t, total_b / (1024^2), total_gc)
    println("=" ^ 95)
end

# ============================================================================
# Parameters
# ============================================================================

const PROP_ORDER = 2
const GREEN_ORDER = 3
const BATCH_SIZE = 5_000_000

model_params = SOE.ModelParams(
    10.0, DATA_DIR, "hopping.txt", "Uijkl.txt",
    "hyb_poles.txt", "hyb_weights.txt"
)
dlr_params = SOE.DLRParams(5.0, 0.01)

println("=" ^ 95)
println("  QMBToolkitSOEHYB End-to-End Benchmark")
println("  3 orbital × 2 spin, 3 poles, Euv=5, rtol=0.01")
println("  Propagator order $(PROP_ORDER), Green order $(GREEN_ORDER)")
println("=" ^ 95)
println()

# ============================================================================
# Shared evaluation logic
# ============================================================================

# NOTE on buffer allocation timing:
# - "buffer alloc" shows the cost of allocate_propagator_buffers/allocate_green_buffers
# - The eval functions (eval_propagator_given_order!, eval_green_given_order!) allocate
#   their OWN buffers internally, so "eval" timing INCLUDES buffer allocation
# - The separate "buffer alloc" timing is informational (shows buffer cost in isolation)
# - To get pure evaluation time, subtract buffer alloc from eval time

function run_propagator!(solver_data, model_data, comm; record_timings = false)
    dlr_grid = model_data.dlr_grid
    for order in 0:PROP_ORDER
        stats_cache = @timed SOE.build_config_cache(
            order, model_data.n_flavors, model_data.hyb.exponents, :propagator_diagram)
        cache = stats_cache[1]
        n_diag = SOE.total_diagrams(cache)
        dims = SOE.config_dimensions(cache)

        # Measure buffer allocation cost separately (informational)
        stats_buf = @timed SOE.allocate_propagator_buffers(solver_data, order)

        # Eval function allocates its own buffers internally
        stats_eval = @timed SOE.eval_propagator_given_order!(
            solver_data, cache, comm; batch_size = BATCH_SIZE)

        # Extract partition function at tau=beta
        prop = SOE.get_propagator(solver_data.calc)
        prop_at_beta = SOE.extract_at_tau(prop.vals, model_data.beta, dlr_grid)
        Z_order = real(sum(diag(prop_at_beta)))
        push!(solver_data.calc.partition_function_by_order, Z_order)
        prop.coeffs = SOE.tau2dlr(dlr_grid, prop.vals, dlr_grid.τ)

        if record_timings
            record!("  prop order $(order): cache build", stats_cache)
            record!("  prop order $(order): buffer alloc (info)", stats_buf)
            record!("  prop order $(order): eval $(n_diag) diags $(dims)", stats_eval)
            # Show per-diagram stats
            if n_diag > 0
                per_diag_ms = stats_eval[2] * 1000 / n_diag
                per_diag_kib = stats_eval[3] / 1024 / n_diag
                @printf("  Order %d: %d diagrams, Z=%.6f, %.2f ms/diag, %.1f KiB/diag\n",
                    order, n_diag, Z_order, per_diag_ms, per_diag_kib)
            else
                @printf("  Order %d: %d diagrams, Z = %.6f\n", order, n_diag, Z_order)
            end
        end
    end
end

function run_green!(solver_data, model_data, comm; record_timings = false)
    Z_final = solver_data.calc.partition_function_by_order[end]
    for order in 1:GREEN_ORDER
        stats_cache = @timed SOE.build_config_cache(
            order, model_data.n_flavors, model_data.hyb.exponents,
            :green_diagram; is_bold = false)
        cache = stats_cache[1]
        n_diag = SOE.total_diagrams(cache)
        dims = SOE.config_dimensions(cache)

        # Measure buffer allocation cost separately (informational)
        stats_buf = @timed SOE.allocate_green_buffers(solver_data, order)

        # Eval function allocates its own buffers internally
        stats_eval = @timed SOE.eval_green_given_order!(
            solver_data, cache, false, comm; batch_size = BATCH_SIZE)

        green_numerator = copy(solver_data.calc.green.vals)
        push!(solver_data.calc.green_numerator_by_order, green_numerator)
        green_order = green_numerator ./ Z_final
        push!(solver_data.calc.green_by_order, green_order)

        if record_timings
            record!("  green order $(order): cache build", stats_cache)
            record!("  green order $(order): buffer alloc (info)", stats_buf)
            record!("  green order $(order): eval $(n_diag) diags $(dims)", stats_eval)
            # Show per-diagram stats
            if n_diag > 0
                per_diag_ms = stats_eval[2] * 1000 / n_diag
                per_diag_kib = stats_eval[3] / 1024 / n_diag
                @printf("  Order %d: %d diagrams, %.2f ms/diag, %.1f KiB/diag\n",
                    order, n_diag, per_diag_ms, per_diag_kib)
            else
                @printf("  Order %d: %d diagrams\n", order, n_diag)
            end
        end
    end
end

# ============================================================================
# 1. Initialization
# ============================================================================

println("--- Initialization ---")

stats = @timed SOE.init_model_data(model_params, dlr_params)
model_data = stats[1]
record!("init_model_data", stats)

stats = @timed SOE.init_solver_data(model_data)
solver_data = stats[1]
record!("init_solver_data", stats)

println("  n_flavors = $(model_data.n_flavors), n_fock = $(model_data.n_fock)")
println("  n_sectors = $(length(SOE.get_bare_propagator(model_data).vals.basis.sectors))")
println("  n_DLR = $(model_data.dlr_grid.size), n_poles = $(length(model_data.hyb.exponents))")
println()

# ============================================================================
# 2. Warmup (JIT compilation)
# ============================================================================

println("--- Warmup (JIT compilation) ---")
t_warmup = @elapsed begin
    run_propagator!(solver_data, model_data, comm)
    run_green!(solver_data, model_data, comm)
end
@printf("  Warmup completed in %.3f s\n", t_warmup)
println()

# Reset solver data for timed run
solver_data = SOE.init_solver_data(model_data)

# ============================================================================
# 3. Timed Propagator Evaluation
# ============================================================================

println("--- Propagator Evaluation (post-warmup) ---")
run_propagator!(solver_data, model_data, comm; record_timings = true)
println()

# ============================================================================
# 4. Timed Green's Function Evaluation
# ============================================================================

println("--- Green's Function Evaluation (post-warmup) ---")
run_green!(solver_data, model_data, comm; record_timings = true)
println()

# ============================================================================
# Summary
# ============================================================================

print_timing_table()
