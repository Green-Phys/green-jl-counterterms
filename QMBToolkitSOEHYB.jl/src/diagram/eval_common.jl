### Evaluate the backbone diagrams to obtain their contributions
using MPI
using LinearAlgebra

"""
    eval_diagrams!(solver_data, solver_params, diagram_type, comm; batch_size=100000)

Evaluate diagrams for a given type using memory-efficient batched generation.

Generates and evaluates diagrams in batches rather than storing all diagrams in memory.
Memory usage is O(batch_size) instead of O(total_diagrams).

# Arguments

  - `solver_data::SolverData{T, Format}`: The solver data
  - `solver_params::SolverParams`: Solver parameters
  - `diagram_type::Symbol`: Type of diagrams (`:propagator_diagram` or `:green_diagram`)
  - `comm::MPI.Comm`: MPI communicator
  - `batch_size::Int`: Number of diagrams per batch (default: 100000)
"""
function eval_diagrams!(solver_data::SolverData{T, Format}, solver_params::SolverParams,
        diagram_type::Symbol, comm::MPI.Comm;
        batch_size::Int = 100000) where {T, Format <: HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    propagator = get_propagator(solver_data.calc)

    if diagram_type == :propagator_diagram && !solver_params.propagator.is_bold
        empty!(solver_data.calc.partition_function_by_order)

        for order in 0:(solver_params.propagator.expansion_order)
            if rank == 0
                cache = build_config_cache(order, solver_data.model.n_flavors,
                    solver_data.model.hyb.exponents, :propagator_diagram)
                dims = config_dimensions(cache)
                println("Evaluating ", dims, " = ", total_diagrams(cache), " propagator diagrams at order ", order)
            end

            eval_propagator_given_order!(solver_data, order, comm; batch_size = batch_size)

            propagator_at_beta = extract_at_tau(
                propagator.vals, solver_data.model.beta, solver_data.model.dlr_grid)
            Z_current = tr(propagator_at_beta)
            push!(solver_data.calc.partition_function_by_order, Z_current)
        end

        propagator.coeffs = tau2dlr(
            solver_data.model.dlr_grid, propagator.vals, solver_data.model.dlr_grid.τ)

    elseif diagram_type == :green_diagram && !solver_params.green.is_bold
        empty!(solver_data.calc.green_numerator_by_order)
        empty!(solver_data.calc.green_by_order)

        for order in 1:(solver_params.green.expansion_order)
            if rank == 0
                cache = build_config_cache(order, solver_data.model.n_flavors,
                    solver_data.model.hyb.exponents, :green_diagram; is_bold = false)
                dims = config_dimensions(cache)
                println("Evaluating ", dims, " = ", total_diagrams(cache), " Green's function diagrams at order ", order)
            end

            eval_green_given_order!(solver_data, order, solver_params.green.is_bold, comm;
                batch_size = batch_size)

            push!(solver_data.calc.green_numerator_by_order,
                DensePropagator{T}(copy(solver_data.calc.green.vals.data)))
        end

        # Compute Green's function by order (numerator / Z)
        for green_order in 1:(solver_params.green.expansion_order)
            propagator_index = green_order
            green_index = green_order
            green_at_order_data = solver_data.calc.green_numerator_by_order[green_index].data ./
                                  solver_data.calc.partition_function_by_order[propagator_index]
            push!(solver_data.calc.green_by_order, DensePropagator{T}(green_at_order_data))
        end

        # Update final Green's function
        copyto!(solver_data.calc.green.vals, solver_data.calc.green_by_order[end])
        solver_data.calc.green.coeffs = tau2dlr(
            solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.τ)
    else
        error("Unknown or unsupported diagram type: $diagram_type")
    end
end

"""
    compute_diagram_prefactor(diagram::BackBone, K0_inverse::Vector{Float64})

Compute the overall prefactor for a diagram.

The prefactor is:
sign * ∏ᵢ (1/K₀ᵢ)^{Kexp_i}

where:

  - sign is the diagram's sign (from topology and crossing parity)
  - K₀ᵢ are the hybridization function values at τ=0
  - Kexp_i are the exponent counts for each SOE component

Only non-zero exponents contribute to the product.
"""
function compute_diagram_prefactor(diagram::BackBone, K0_inverse::Vector{Float64})
    prefactor = diagram.sign
    for (idx, Kexp) in enumerate(diagram.prefactor_Kexps)
        if Kexp != 0
            prefactor *= K0_inverse[idx]^Kexp
        end
    end
    return prefactor
end
