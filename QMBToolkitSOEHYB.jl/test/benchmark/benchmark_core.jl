# benchmark_core.jl - Micro-benchmarks for individual hot-path operations.
#
# Benchmarks individual operations on a 3-orbital × 2-spin (6 spin-orbital)
# model with 3 SOE poles at propagator order 2 / Green order 3.
#
# Run with:
#     julia --project=. test/benchmark/benchmark_core.jl
#
# Requires BenchmarkTools.jl:
#     julia -e 'import Pkg; Pkg.add("BenchmarkTools")'

using QMBToolkitSOEHYB
using MPI
using Printf

const SOE = QMBToolkitSOEHYB

# Check for BenchmarkTools
try
    @eval using BenchmarkTools
catch
    error("BenchmarkTools.jl is required. Install with: import Pkg; Pkg.add(\"BenchmarkTools\")")
end

# ============================================================================
# Setup
# ============================================================================

MPI.Init()

const BENCHMARK_DIR = @__DIR__
const DATA_DIR = joinpath(BENCHMARK_DIR, "data")

println("=" ^ 80)
println("  QMBToolkitSOEHYB Core Micro-Benchmarks")
println("=" ^ 80)
println()

# Initialize model (3 orbital × 2 spin = 6 spin-orbitals, 2^6 = 64 Fock states)
model_params = SOE.ModelParams(
    10.0, DATA_DIR, "hopping.txt", "Uijkl.txt",
    "hyb_poles.txt", "hyb_weights.txt"
)
dlr_params = SOE.DLRParams(5.0, 0.01)

println("Initializing model data...")
model_data = SOE.init_model_data(model_params, dlr_params)
solver_data = SOE.init_solver_data(model_data)

println("  n_flavors          = $(model_data.n_flavors)")
println("  n_flavors_physical = $(model_data.n_flavors_physical)")
println("  n_fock             = $(model_data.n_fock)")
println("  n_DLR              = $(model_data.dlr_grid.size)")
println("  n_poles (p)        = $(length(model_data.hyb.exponents))")

basis = SOE.get_bare_propagator(model_data).vals.basis
n_sectors = length(basis.sectors)
sector_dims = [s.dim for s in basis.sectors]
println("  n_sectors          = $(n_sectors)")
println("  sector_dims        = $(sector_dims)")
println()

# Extract commonly used objects
dlr_grid = model_data.dlr_grid
n_tau = length(dlr_grid.τ)
n_dlr = dlr_grid.size
bare_prop = SOE.get_bare_propagator(model_data)
zero_flux = bare_prop.vals.flux
conv_ws = model_data.convolution_workspace
kernel_cache = model_data.kernel_cache
dlr_ws = conv_ws.dlr_workspace

ann_opts = SOE.get_ann_opts(model_data)
cre_opts = SOE.get_cre_opts(model_data)

println("  n_tau              = $(n_tau)")
println("  n_dlr              = $(n_dlr)")
println()

# ============================================================================
# Formatting helpers
# ============================================================================

function print_bench(name::String, trial)
    med = median(trial)
    t_us = med.time / 1e3
    allocs = med.allocs
    mem_kib = med.memory / 1024
    if t_us >= 1000.0
        @printf("  %-55s  %8.2f ms  %6d allocs  %8.1f KiB\n", name, t_us / 1000, allocs, mem_kib)
    else
        @printf("  %-55s  %8.2f μs  %6d allocs  %8.1f KiB\n", name, t_us, allocs, mem_kib)
    end
end

function section_header(name::String)
    println("-" ^ 80)
    println("  ", name)
    println("-" ^ 80)
end

# ============================================================================
# 1. Hybridization Application
# ============================================================================

section_header("1. Hybridization Application")

let
    prop_source = bare_prop.vals
    prop_working = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)
    hyb_kp = SOE.KPlus(1)
    hyb_km = SOE.KMinus(1)

    b = @benchmark SOE.apply_hybridization_inplace!(
        $prop_working, $hyb_kp, $model_data
    ) evals=1 setup=(copyto!($prop_working, $prop_source))
    print_bench("apply_hybridization_inplace! (KPlus)", b)

    b = @benchmark SOE.apply_hybridization_inplace!(
        $prop_working, $hyb_km, $model_data
    ) evals=1 setup=(copyto!($prop_working, $prop_source))
    print_bench("apply_hybridization_inplace! (KMinus)", b)
end
println()

# ============================================================================
# 2. Operator-Propagator Multiplications
# ============================================================================

section_header("2. Operator-Propagator Multiplications")

let
    prop = bare_prop.vals

    # op * prop (creation operator × propagator)
    op = cre_opts[1]
    result_flux = SOE.compute_new_flux(zero_flux, op)
    result = SOE.allocate_buffer_for_flux(basis, result_flux, n_tau, ComplexF64)

    b = @benchmark SOE.multiply_op_prop!($result, $op, $prop)
    print_bench("multiply_op_prop! (cre × prop)", b)

    # prop * op
    result2 = SOE.allocate_buffer_for_flux(basis, result_flux, n_tau, ComplexF64)
    b = @benchmark SOE.multiply_prop_op!($result2, $prop, $op)
    print_bench("multiply_prop_op! (prop × cre)", b)

    # prop * prop (both zero flux)
    result3 = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)
    b = @benchmark SOE.multiply_prop_prop!($result3, $prop, $prop)
    print_bench("multiply_prop_prop! (prop × prop)", b)

    # trace
    trace_buf = zeros(Float64, n_tau)
    b = @benchmark SOE.trace_prop!($trace_buf, $prop)
    print_bench("trace_prop!", b)
end
println()

# ============================================================================
# 3. DLR Transformations
# ============================================================================

section_header("3. DLR Transformations")

let
    src = bare_prop.vals
    dest_dlr = SOE.allocate_buffer_for_flux(basis, zero_flux, n_dlr, ComplexF64)
    tau_grid = dlr_grid.τ

    # In-place tau2dlr!
    b = @benchmark SOE.tau2dlr!($dest_dlr, $dlr_grid, $src, $tau_grid, $dlr_ws)
    print_bench("tau2dlr! (Propagator, in-place)", b)

    # Allocating tau2dlr
    b = @benchmark SOE.tau2dlr($dlr_grid, $src, $tau_grid)
    print_bench("tau2dlr (Propagator, allocating)", b)

    # dlr2tau (allocating)
    prop_dlr = SOE.tau2dlr(dlr_grid, src, tau_grid)
    b = @benchmark SOE.dlr2tau($dlr_grid, $prop_dlr, $tau_grid)
    print_bench("dlr2tau (Propagator, allocating)", b)
end
println()

# ============================================================================
# 4. Convolution
# ============================================================================

section_header("4. Convolution (do_convolution_dlr2tau!)")

let
    src = bare_prop.vals
    F_dlr = SOE.tau2dlr(dlr_grid, src, dlr_grid.τ)
    G_dlr = SOE.tau2dlr(dlr_grid, src, dlr_grid.τ)
    dest = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)

    b = @benchmark SOE.do_convolution_dlr2tau!(
        $dest, $conv_ws, $dlr_grid, $kernel_cache, $F_dlr, $G_dlr
    )
    print_bench("do_convolution_dlr2tau! (Propagator)", b)
end
println()

# ============================================================================
# 5. Reflection
# ============================================================================

section_header("5. Reflection (do_reflection_tau2tau!)")

let
    src = bare_prop.vals
    dest = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)
    dlr_buffer = SOE.allocate_buffer_for_flux(basis, zero_flux, n_dlr, ComplexF64)
    τ_reflected = conv_ws.τ_reflected
    kernel_tau_reflected = conv_ws.kernel_tau_reflected

    b = @benchmark SOE.do_reflection_tau2tau!(
        $dest, $dlr_buffer, $dlr_grid, $src, $dlr_ws;
        τ_reflected = $τ_reflected
    )
    print_bench("do_reflection_tau2tau! (no cache)", b)

    b = @benchmark SOE.do_reflection_tau2tau!(
        $dest, $dlr_buffer, $dlr_grid, $src, $dlr_ws;
        τ_reflected = $τ_reflected, kernel_tau_reflected = $kernel_tau_reflected
    )
    print_bench("do_reflection_tau2tau! (cached)", b)
end
println()

# ============================================================================
# 6. Full Backbone Evaluation
# ============================================================================

section_header("6. Full Backbone Evaluation")

# Propagator order 2 (L=2 hyb lines, M=6 nodes)
let
    cache = SOE.build_config_cache(
        2, model_data.n_flavors, model_data.hyb.exponents, :propagator_diagram)
    n_diag = SOE.total_diagrams(cache)
    println("  Propagator order 2: $(n_diag) diagrams")

    diagram = SOE.generate_diagram_at_index(cache, 1)
    buffers = SOE.allocate_propagator_buffers(solver_data, 2)

    b = @benchmark SOE.eval_propagator_forward_backbone($diagram, $solver_data, $buffers) evals=1
    print_bench("eval_propagator_forward_backbone (order 2)", b)

    # Also benchmark multiplication_convolution directly
    nodes = diagram.nodes
    edges = diagram.edges
    prop_vals = bare_prop.vals
    b = @benchmark SOE.multiplication_convolution(
        $nodes, $edges, $dlr_grid, $model_data, $prop_vals, $buffers
    ) evals=1
    print_bench("multiplication_convolution (order 2)", b)
end

println()

# Green's function order 3 (L=2 hyb lines, M=7 nodes)
# NOTE: Different diagrams have different split_points, affecting forward/reverse work distribution.
# We sample multiple diagrams to show the variation.
let
    cache_g = SOE.build_config_cache(
        3, model_data.n_flavors, model_data.hyb.exponents,
        :green_diagram; is_bold = false)
    n_diag = SOE.total_diagrams(cache_g)
    println("  Green order 3: $(n_diag) diagrams")

    buffers_g = SOE.allocate_green_buffers(solver_data, 3)
    prop_vals = bare_prop.vals

    # Sample 5 diagrams across the range to show split_point variation
    sample_indices = [1, n_diag ÷ 4, n_diag ÷ 2, 3 * n_diag ÷ 4, n_diag]
    println("  Sampling diagrams at indices: $(sample_indices)")
    println("  (Split point determines forward/reverse work distribution)")

    for (i, idx) in enumerate(sample_indices)
        diagram_g = SOE.generate_diagram_at_index(cache_g, idx)
        split_pt = diagram_g.external_opts_loc[2]
        n_nodes = length(diagram_g.nodes)
        fwd_iters = max(0, split_pt - 2)  # forward loop iterations
        rev_iters = max(0, n_nodes - split_pt - 1)  # reverse loop iterations

        # Warmup
        SOE.eval_green_forward_backbone(diagram_g, split_pt, solver_data, prop_vals, buffers_g)
        SOE.eval_green_reverse_backbone(diagram_g, split_pt, solver_data, prop_vals, buffers_g)

        b_fwd = @benchmark SOE.eval_green_forward_backbone(
            $diagram_g, $split_pt, $solver_data, $prop_vals, $buffers_g
        ) evals=1
        b_rev = @benchmark SOE.eval_green_reverse_backbone(
            $diagram_g, $split_pt, $solver_data, $prop_vals, $buffers_g
        ) evals=1

        fwd_time = median(b_fwd).time / 1e3
        rev_time = median(b_rev).time / 1e3
        @printf("    Diagram %d: split=%d, fwd_iters=%d (%.1f μs), rev_iters=%d (%.1f μs)\n",
            idx, split_pt, fwd_iters, fwd_time, rev_iters, rev_time)
    end
    println()

    # Also show aggregate stats using diagram #1 for multiplication_convolution_reverse
    diagram_g = SOE.generate_diagram_at_index(cache_g, 1)
    split_pt = diagram_g.external_opts_loc[2]
    rev_nodes = @view(diagram_g.nodes[split_pt:end])
    rev_edges = @view(diagram_g.edges[split_pt:end])
    rev_buffers = buffers_g.rev_buffers
    b = @benchmark SOE.multiplication_convolution_reverse(
        $rev_nodes, $rev_edges, $dlr_grid, $model_data, $prop_vals, $rev_buffers
    ) evals=1
    print_bench("multiplication_convolution_reverse (order 3, diag 1)", b)
end
println()

# ============================================================================
# 7. Diagram Generation
# ============================================================================

section_header("7. Diagram Generation")

let
    cache_p2 = SOE.build_config_cache(
        2, model_data.n_flavors, model_data.hyb.exponents, :propagator_diagram)
    cache_g3 = SOE.build_config_cache(
        3, model_data.n_flavors, model_data.hyb.exponents,
        :green_diagram; is_bold = false)

    total_p2 = SOE.total_diagrams(cache_p2)
    total_g3 = SOE.total_diagrams(cache_g3)
    println("  Propagator order 2: $(total_p2) total diagrams")
    println("  Green order 3:      $(total_g3) total diagrams")

    # Single diagram generation
    b = @benchmark SOE.generate_diagram_at_index($cache_p2, 1)
    print_bench("generate_diagram_at_index (prop order 2)", b)

    b = @benchmark SOE.generate_diagram_at_index($cache_g3, 1)
    print_bench("generate_diagram_at_index (green order 3)", b)

    # Batch generation (100 diagrams)
    batch_p = min(100, total_p2)
    b = @benchmark SOE.generate_diagram_batch($cache_p2, 1, $batch_p)
    print_bench("generate_diagram_batch (prop order 2, $(batch_p) diags)", b)

    batch_g = min(100, total_g3)
    b = @benchmark SOE.generate_diagram_batch($cache_g3, 1, $batch_g)
    print_bench("generate_diagram_batch (green order 3, $(batch_g) diags)", b)

    # Config cache build
    n_flav = model_data.n_flavors
    exps = model_data.hyb.exponents
    b = @benchmark SOE.build_config_cache(2, $n_flav, $exps, :propagator_diagram)
    print_bench("build_config_cache (prop order 2)", b)

    b = @benchmark SOE.build_config_cache(
        3, $n_flav, $exps, :green_diagram; is_bold = false)
    print_bench("build_config_cache (green order 3)", b)
end
println()

# ============================================================================
# 8. Buffer Allocation
# ============================================================================

section_header("8. Buffer Allocation")

let
    b = @benchmark SOE.allocate_propagator_buffers($solver_data, 2)
    print_bench("allocate_propagator_buffers (order 2)", b)

    b = @benchmark SOE.allocate_propagator_buffers($solver_data, 3)
    print_bench("allocate_propagator_buffers (order 3)", b)

    b = @benchmark SOE.allocate_green_buffers($solver_data, 2)
    print_bench("allocate_green_buffers (order 2)", b)

    b = @benchmark SOE.allocate_green_buffers($solver_data, 3)
    print_bench("allocate_green_buffers (order 3)", b)
end
println()

# ============================================================================
# 9. Per-Call Allocation Measurement
# ============================================================================

section_header("9. Per-Call Allocation Analysis")

println("  Measuring allocation bytes per call for optimization targets:")
println()

let
    # Propagator convolution
    F_dlr = SOE.tau2dlr(dlr_grid, bare_prop.vals, dlr_grid.τ)
    G_dlr = SOE.tau2dlr(dlr_grid, bare_prop.vals, dlr_grid.τ)
    dest = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)

    # Warmup
    SOE.do_convolution_dlr2tau!(dest, conv_ws, dlr_grid, kernel_cache, F_dlr, G_dlr)

    stats = @timed SOE.do_convolution_dlr2tau!(dest, conv_ws, dlr_grid, kernel_cache, F_dlr, G_dlr)
    @printf("  do_convolution_dlr2tau! (propagator):  %6d bytes, %.1f μs\n", stats[3], stats[2] * 1e6)

    # Reflection (without cached kernel)
    dest_refl = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)
    dlr_buf = SOE.allocate_buffer_for_flux(basis, zero_flux, n_dlr, ComplexF64)
    τ_reflected = conv_ws.τ_reflected

    # Warmup
    SOE.do_reflection_tau2tau!(dest_refl, dlr_buf, dlr_grid, bare_prop.vals, dlr_ws; τ_reflected = τ_reflected)

    stats = @timed SOE.do_reflection_tau2tau!(dest_refl, dlr_buf, dlr_grid, bare_prop.vals, dlr_ws; τ_reflected = τ_reflected)
    @printf("  do_reflection_tau2tau! (no cache):     %6d bytes, %.1f μs\n", stats[3], stats[2] * 1e6)

    # Reflection (with cached kernel - S5 optimization)
    kernel_tau_reflected = conv_ws.kernel_tau_reflected
    SOE.do_reflection_tau2tau!(dest_refl, dlr_buf, dlr_grid, bare_prop.vals, dlr_ws;
        τ_reflected = τ_reflected, kernel_tau_reflected = kernel_tau_reflected)
    stats = @timed SOE.do_reflection_tau2tau!(dest_refl, dlr_buf, dlr_grid, bare_prop.vals, dlr_ws;
        τ_reflected = τ_reflected, kernel_tau_reflected = kernel_tau_reflected)
    @printf("  do_reflection_tau2tau! (cached):       %6d bytes, %.1f μs\n", stats[3], stats[2] * 1e6)

    # tau2dlr (in-place)
    dest_dlr = SOE.allocate_buffer_for_flux(basis, zero_flux, n_dlr, ComplexF64)
    SOE.tau2dlr!(dest_dlr, dlr_grid, bare_prop.vals, dlr_grid.τ, dlr_ws)
    stats = @timed SOE.tau2dlr!(dest_dlr, dlr_grid, bare_prop.vals, dlr_grid.τ, dlr_ws)
    @printf("  tau2dlr! (in-place):                   %6d bytes, %.1f μs\n", stats[3], stats[2] * 1e6)

    # Operator multiply
    op = cre_opts[1]
    result_flux = SOE.compute_new_flux(zero_flux, op)
    result_op = SOE.allocate_buffer_for_flux(basis, result_flux, n_tau, ComplexF64)
    SOE.multiply_op_prop!(result_op, op, bare_prop.vals)
    stats = @timed SOE.multiply_op_prop!(result_op, op, bare_prop.vals)
    @printf("  multiply_op_prop!:                     %6d bytes, %.1f μs\n", stats[3], stats[2] * 1e6)

    # Hybridization
    prop_copy = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)
    copyto!(prop_copy, bare_prop.vals)
    hyb_kp = SOE.KPlus(1)
    SOE.apply_hybridization_inplace!(prop_copy, hyb_kp, model_data)
    copyto!(prop_copy, bare_prop.vals)
    stats = @timed SOE.apply_hybridization_inplace!(prop_copy, hyb_kp, model_data)
    @printf("  apply_hybridization_inplace!:          %6d bytes, %.1f μs\n", stats[3], stats[2] * 1e6)
end
println()

# Flux enumeration analysis
section_header("10. Flux Enumeration Analysis (M1 Optimization)")

let
    QN = typeof(basis.sectors[1].qn)
    println("  Analyzing valid flux values for QN_NSz basis with $(length(basis.sectors)) sectors:")
    println()

    valid_fluxes = SOE.compute_valid_fluxes(basis)
    @printf("    Valid fluxes (used for buffer allocation): %d\n", length(valid_fluxes))
    println()

    # Show theoretical maximum range for comparison
    # For n_orbitals=3 with 2 spins, max |ΔN| = 6, so theoretical max for order 3 would be
    # much larger than the actual valid set
    n_sectors = length(basis.sectors)
    println("    Basis has $n_sectors sectors, so maximum |ΔN| = $(n_sectors-1)")
    println("    Only fluxes with valid (target, source) sector pairs are allocated.")
    println()
    println("  M1 optimization: Using compute_valid_fluxes instead of enumerate_possible_flux")
    println("  - Reduces buffer allocation by only allocating for valid fluxes")
    println("  - Early exit when invalid flux detected (diagram contributes zero)")
end
println()

# ============================================================================
# Summary
# ============================================================================

println("=" ^ 80)
println("  Benchmarks complete.")
println("=" ^ 80)
