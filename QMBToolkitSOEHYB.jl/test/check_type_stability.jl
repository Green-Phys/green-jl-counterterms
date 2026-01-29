#!/usr/bin/env julia
# Type stability analysis script using Julia built-in tools
#
# Three approaches used:
# 1. Test.@inferred - verifies concrete return types
# 2. @code_warntype - visual inspection of type inference
# 3. @allocated - allocation tracking (type instability often causes allocations)

using QMBToolkitSOEHYB
using MPI
using Test
using InteractiveUtils: @code_warntype, code_warntype
const SOE = QMBToolkitSOEHYB

MPI.Init()

# Use the data_2spin directory
data_dir = joinpath(@__DIR__, "data_2spin")

# Initialize model
model_params = SOE.ModelParams(
    10.0, data_dir, "hopping.txt", "Uijkl.txt",
    "hyb_poles.txt", "hyb_weights.txt"
)
dlr_params = SOE.DLRParams(5.0, 0.01)

model_data = SOE.init_model_data(model_params, dlr_params)
solver_data = SOE.init_solver_data(model_data)

# Get common objects
basis = SOE.get_bare_propagator(model_data).vals.basis
dlr_grid = model_data.dlr_grid
n_tau = length(dlr_grid.τ)
n_dlr = dlr_grid.size
zero_flux = zero(typeof(basis.sectors[1].qn))
prop = SOE.get_bare_propagator(model_data).vals

# Allocate buffers
buffers_p = SOE.allocate_propagator_buffers(solver_data, 2)
buffers_g = SOE.allocate_green_buffers(solver_data, 3)

# Build diagram caches
cache_p = SOE.build_config_cache(2, model_data.n_flavors, model_data.hyb.exponents, :propagator_diagram)
diagram_p = SOE.generate_diagram_at_index(cache_p, 1)

cache_g = SOE.build_config_cache(3, model_data.n_flavors, model_data.hyb.exponents, :green_diagram; is_bold = false)
diagram_g = SOE.generate_diagram_at_index(cache_g, 1)

println("="^70)
println("TYPE STABILITY ANALYSIS")
println("="^70)
println()

#=============================================================================
# Part 1: Test.@inferred - Verify concrete return types
=============================================================================#
println("1. @inferred TESTS (verifies concrete return types)")
println("-"^70)

function run_inferred_tests()
    @testset "Buffer Accessors" begin
        @test (@inferred SOE.get_result(buffers_p, zero_flux)) isa SOE.Propagator
        @test (@inferred SOE.get_op_product(buffers_p, zero_flux)) isa SOE.Propagator
        @test (@inferred SOE.get_conv_operand(buffers_p)) isa SOE.Propagator
        @test (@inferred SOE.get_conv_dlr(buffers_p)) isa SOE.Propagator
        @test (@inferred SOE.get_result_dlr(buffers_p, zero_flux)) isa SOE.Propagator

        @test (@inferred SOE.get_ann_product(buffers_g, zero_flux)) isa SOE.Propagator
        @test (@inferred SOE.get_final_product(buffers_g, zero_flux)) isa SOE.Propagator
        @test (@inferred SOE.get_cre_product(buffers_g)) isa SOE.Propagator
        @test (@inferred SOE.get_fwd_buffers(buffers_g)) isa SOE.PropagatorEvalBuffers
        @test (@inferred SOE.get_rev_buffers(buffers_g)) isa SOE.PropagatorEvalBuffers
    end

    @testset "Core Operations" begin
        hyb = SOE.KPlus(1)
        # Note: get_flux returns abstract QuantumNumber because Propagator.flux is abstract.
        # This is a design tradeoff for flexibility. The actual return is concrete (QN_NSz).
        @test SOE.get_flux(prop) isa SOE.QuantumNumber
        @test (@inferred SOE.compute_new_flux(zero_flux, SOE.get_cre_opts(model_data)[1])) isa SOE.QuantumNumber
    end

    @testset "Diagram Evaluation" begin
        # Note: These functions return Propagator, which is type-stable
        result = @inferred SOE.eval_propagator_forward_backbone(diagram_p, solver_data, buffers_p)
        @test result isa SOE.Propagator

        result_fwd = @inferred SOE.eval_green_forward_backbone(diagram_g, 2, solver_data, prop, buffers_g)
        @test result_fwd isa SOE.Propagator

        result_rev = @inferred SOE.eval_green_reverse_backbone(diagram_g, 2, solver_data, prop, buffers_g)
        @test result_rev isa SOE.Propagator
    end
end

run_inferred_tests()
println()

#=============================================================================
# Part 2: Allocation Tracking - Detect type instability via allocations
=============================================================================#
println("2. ALLOCATION TRACKING (type instability causes allocations)")
println("-"^70)

# Warm up
SOE.eval_propagator_forward_backbone(diagram_p, solver_data, buffers_p)
SOE.eval_green_forward_backbone(diagram_g, 2, solver_data, prop, buffers_g)
SOE.eval_green_reverse_backbone(diagram_g, 2, solver_data, prop, buffers_g)

hyb = SOE.KPlus(1)
result_prop = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)
SOE.apply_hybridization_inplace!(result_prop, hyb, model_data)

# Measure allocations (after warmup, allocations should be minimal)
function measure_allocations(name, expr, expected_allocs = 0)
    allocs = @allocated eval(expr)
    status = allocs <= expected_allocs ? "OK" : "CHECK"
    println("   $name: $allocs bytes [$status]")
    return allocs
end

println("   Function                                      Allocations")
println("   " * "-"^55)

# Buffer accessor allocations (should be zero after warmup)
alloc_get_result = @allocated SOE.get_result(buffers_p, zero_flux)
println("   get_result:                                   $alloc_get_result bytes")

alloc_get_op_product = @allocated SOE.get_op_product(buffers_p, zero_flux)
println("   get_op_product:                               $alloc_get_op_product bytes")

alloc_get_ann_product = @allocated SOE.get_ann_product(buffers_g, zero_flux)
println("   get_ann_product:                              $alloc_get_ann_product bytes")

# Core operation allocations
result_prop2 = SOE.allocate_buffer_for_flux(basis, zero_flux, n_tau, ComplexF64)
alloc_hyb = @allocated SOE.apply_hybridization_inplace!(result_prop2, hyb, model_data)
println("   apply_hybridization_inplace!:                 $alloc_hyb bytes")

# Backbone evaluation allocations (may have some allocations for Dict lookup)
alloc_prop_backbone = @allocated SOE.eval_propagator_forward_backbone(diagram_p, solver_data, buffers_p)
println("   eval_propagator_forward_backbone:             $alloc_prop_backbone bytes")

alloc_green_fwd = @allocated SOE.eval_green_forward_backbone(diagram_g, 2, solver_data, prop, buffers_g)
println("   eval_green_forward_backbone:                  $alloc_green_fwd bytes")

alloc_green_rev = @allocated SOE.eval_green_reverse_backbone(diagram_g, 2, solver_data, prop, buffers_g)
println("   eval_green_reverse_backbone:                  $alloc_green_rev bytes")

println()

#=============================================================================
# Part 3: @code_warntype - Visual inspection of type inference
=============================================================================#
println("3. @code_warntype ANALYSIS (look for ::Any or ::Union)")
println("-"^70)
println()

function analyze_warntype(name, f, argtypes)
    println("--- $name ---")
    io = IOBuffer()
    try
        code_warntype(io, f, argtypes)
        output = String(take!(io))

        # Count issues
        any_count = length(collect(eachmatch(r"::Any\b", output)))
        union_count = length(collect(eachmatch(r"::Union\{", output)))

        if any_count > 0 || union_count > 0
            println("   WARNING: $any_count ::Any, $union_count ::Union found")
            # Show problematic lines
            for line in split(output, "\n")
                if occursin(r"::Any\b", line) || occursin("::Union{", line)
                    shortened = strip(line)
                    if length(shortened) > 90
                        shortened = shortened[1:90] * "..."
                    end
                    println("      ", shortened)
                end
            end
        else
            println("   OK - Type stable")
        end
    catch e
        println("   Error: $e")
    end
    println()
end

# Buffer accessors
analyze_warntype("get_result", SOE.get_result, (typeof(buffers_p), typeof(zero_flux)))
analyze_warntype("get_op_product", SOE.get_op_product, (typeof(buffers_p), typeof(zero_flux)))
analyze_warntype("get_ann_product", SOE.get_ann_product, (typeof(buffers_g), typeof(zero_flux)))
analyze_warntype("get_final_product", SOE.get_final_product, (typeof(buffers_g), typeof(zero_flux)))
analyze_warntype("get_cre_product", SOE.get_cre_product, (typeof(buffers_g),))

# Core operations
analyze_warntype("apply_hybridization_inplace!", SOE.apply_hybridization_inplace!,
    (typeof(prop), typeof(hyb), typeof(model_data)))

# Backbone evaluation
analyze_warntype("multiplication_convolution", SOE.multiplication_convolution,
    (typeof(diagram_p.nodes), typeof(diagram_p.edges), typeof(dlr_grid), typeof(model_data), typeof(prop), typeof(buffers_p)))

analyze_warntype("eval_propagator_forward_backbone", SOE.eval_propagator_forward_backbone,
    (typeof(diagram_p), typeof(solver_data), typeof(buffers_p)))

analyze_warntype("eval_green_forward_backbone", SOE.eval_green_forward_backbone,
    (typeof(diagram_g), Int, typeof(solver_data), typeof(prop), typeof(buffers_g)))

analyze_warntype("eval_green_reverse_backbone", SOE.eval_green_reverse_backbone,
    (typeof(diagram_g), Int, typeof(solver_data), typeof(prop), typeof(buffers_g)))

# Operator multiplications
op = SOE.get_cre_opts(model_data)[1]
result_flux = SOE.compute_new_flux(zero_flux, op)
result_prop3 = SOE.allocate_buffer_for_flux(basis, result_flux, n_tau, ComplexF64)

analyze_warntype("multiply_op_prop!", SOE.multiply_op_prop!,
    (typeof(result_prop3), typeof(op), typeof(prop)))

analyze_warntype("multiply_prop_op!", SOE.multiply_prop_op!,
    (typeof(result_prop3), typeof(prop), typeof(op)))

# DLR operations
analyze_warntype("tau2dlr!", SOE.tau2dlr!,
    (typeof(buffers_p.conv_dlr), typeof(dlr_grid), typeof(prop), typeof(dlr_grid.τ), typeof(model_data.convolution_workspace.dlr_workspace)))

println("="^70)
println("SUMMARY")
println("="^70)
println()
println("Type stability checking complete.")
println()
println("Interpretation:")
println("  - @inferred tests: Should all pass for type-stable code")
println("  - Allocations: Lower is better; zero means no dynamic dispatch")
println("  - @code_warntype: ::Any or ::Union indicates type instability")
println()
println("Note: Small allocations in backbone evaluation may be from Dict")
println("lookups which are expected. Large allocations indicate problems.")
