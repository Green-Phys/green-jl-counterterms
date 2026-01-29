# profile_solver.jl - Profile.jl script to generate flame graph data.
#
# Profiles the solver on the 3-orbital × 2-spin benchmark case.
# Outputs a flat profile to stdout and saves profile data for flame graph tools.
#
# Run with:
#     julia --project=. test/benchmark/profile_solver.jl
#
# For flame graphs, install PProf.jl:
#     julia -e 'import Pkg; Pkg.add("PProf")'
# Then view at http://localhost:57599 after the script completes.

using QMBToolkitSOEHYB
using MPI
using Profile
using Serialization
using LinearAlgebra: diag

const SOE = QMBToolkitSOEHYB

MPI.Init()
comm = MPI.COMM_WORLD

const BENCHMARK_DIR = @__DIR__
const DATA_DIR = joinpath(BENCHMARK_DIR, "data")
const BATCH_SIZE = 5_000_000
const PROP_ORDER = 2
const GREEN_ORDER = 3

# ============================================================================
# Setup
# ============================================================================

model_params = SOE.ModelParams(
    10.0, DATA_DIR, "hopping.txt", "Uijkl.txt",
    "hyb_poles.txt", "hyb_weights.txt"
)
dlr_params = SOE.DLRParams(5.0, 0.01)

println("Initializing model...")
model_data = SOE.init_model_data(model_params, dlr_params)
solver_data = SOE.init_solver_data(model_data)

println("  n_flavors=$(model_data.n_flavors), n_fock=$(model_data.n_fock), " *
        "n_DLR=$(model_data.dlr_grid.size), n_poles=$(length(model_data.hyb.exponents))")
println()

# ============================================================================
# Warmup: run the full evaluation once to compile everything
# ============================================================================

println("Warming up (first run to trigger JIT)...")

function run_evaluation!(solver_data, model_data, comm)
    dlr_grid = model_data.dlr_grid

    # Propagator evaluation
    for order in 0:PROP_ORDER
        cache = SOE.build_config_cache(
            order, model_data.n_flavors, model_data.hyb.exponents, :propagator_diagram)
        SOE.eval_propagator_given_order!(solver_data, cache, comm; batch_size = BATCH_SIZE)

        prop = SOE.get_propagator(solver_data.calc)
        prop_at_beta = SOE.extract_at_tau(prop.vals, model_data.beta, dlr_grid)
        Z = real(sum(diag(prop_at_beta)))
        push!(solver_data.calc.partition_function_by_order, Z)
        prop.coeffs = SOE.tau2dlr(dlr_grid, prop.vals, dlr_grid.τ)
    end

    # Green's function evaluation
    for order in 1:GREEN_ORDER
        cache = SOE.build_config_cache(
            order, model_data.n_flavors, model_data.hyb.exponents,
            :green_diagram; is_bold = false)
        SOE.eval_green_given_order!(solver_data, cache, false, comm; batch_size = BATCH_SIZE)
    end
end

# Warmup run
run_evaluation!(solver_data, model_data, comm)
println("Warmup complete.")
println()

# ============================================================================
# Reset solver data for profiled run
# ============================================================================

solver_data = SOE.init_solver_data(model_data)

# ============================================================================
# Profile the evaluation
# ============================================================================

println("Profiling evaluation (second run, JIT-free)...")
Profile.clear()
Profile.init(n = 10_000_000, delay = 0.0001)  # 100μs sampling, large buffer

@profile run_evaluation!(solver_data, model_data, comm)

println()
println("=" ^ 80)
println("  Profile: Flat (top 40 by count)")
println("=" ^ 80)
Profile.print(maxdepth = 20, mincount = 5, noisefloor = 2.0, sortedby = :count)

println()
println("=" ^ 80)
println("  Profile: Tree")
println("=" ^ 80)
Profile.print(format = :tree, maxdepth = 12, mincount = 5, noisefloor = 2.0)

# ============================================================================
# Save profile data for external tools
# ============================================================================

profile_file = joinpath(BENCHMARK_DIR, "profile_data.bin")
open(profile_file, "w") do io
    serialize(io, Profile.retrieve())
end
println()
println("Profile data saved to: $(profile_file)")

# Try PProf if available
try
    @eval using PProf
    pprof_file = joinpath(BENCHMARK_DIR, "profile.pb.gz")
    pprof(out = pprof_file, web = false)
    println("PProf data saved to: $(pprof_file)")
    println("View with: julia -e 'using PProf; PProf.refresh(file=\"$(pprof_file)\")' ")
catch
    println("(PProf.jl not installed -- skipping flame graph export)")
    println("Install with: import Pkg; Pkg.add(\"PProf\")")
end

println()
println("Done.")
