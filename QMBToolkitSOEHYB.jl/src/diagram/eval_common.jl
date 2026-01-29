### Evaluate the backbone diagrams to obtain their contributions
using MPI
using LinearAlgebra
using QMBToolkitDLR

"""
    eval_diagrams!(solver_data, solver_params, diagram_type, comm)

Evaluate diagrams for a given type using zero-allocation diagram generation.

Diagrams are generated on-the-fly using a single reusable MutableBackBone,
eliminating per-diagram allocation overhead.

# Arguments

  - `solver_data::SolverData{T}`: The solver data
  - `solver_params::SolverParams`: Solver parameters
  - `diagram_type::Symbol`: Type of diagrams (`:propagator_diagram` or `:green_diagram`)
  - `comm::MPI.Comm`: MPI communicator
"""
function eval_diagrams!(solver_data::SolverData{T}, solver_params::SolverParams,
        diagram_type::Symbol, comm::MPI.Comm) where {T}
    rank = MPI.Comm_rank(comm)
    propagator = get_propagator(solver_data.calc)

    if diagram_type == :propagator_diagram && !solver_params.propagator.is_bold
        empty!(solver_data.calc.partition_function_by_order)
        empty!(solver_data.calc.propagator_diagram_info)

        for order in 0:(solver_params.propagator.expansion_order)
            cache = build_config_cache(order, solver_data.model.n_flavors,
                solver_data.model.hyb.exponents, :propagator_diagram)
            dims = config_dimensions(cache)
            n_total = total_diagrams(cache)

            if rank == 0
                println("Evaluating ", dims, " = ", n_total, " propagator diagrams at order ", order)
            end

            # Store diagram info: (n_topologies, n_directions, n_flavors_combo, n_exponents_combo, total)
            push!(solver_data.calc.propagator_diagram_info, (dims..., n_total))

            eval_propagator_given_order!(solver_data, cache, comm)

            propagator_at_beta = extract_at_tau(
                propagator.vals, solver_data.model.beta, solver_data.model.dlr_grid)
            Z_current = tr(propagator_at_beta)
            # Use real() since partition function must be real; imaginary part is numerical noise
            push!(solver_data.calc.partition_function_by_order, real(Z_current))
        end

        propagator.coeffs = tau2dlr(
            solver_data.model.dlr_grid, propagator.vals, solver_data.model.dlr_grid.τ)

    elseif diagram_type == :green_diagram && !solver_params.green.is_bold
        empty!(solver_data.calc.green_numerator_by_order)
        empty!(solver_data.calc.green_by_order)
        empty!(solver_data.calc.green_diagram_info)

        for order in 1:(solver_params.green.expansion_order)
            cache = build_config_cache(order, solver_data.model.n_flavors,
                solver_data.model.hyb.exponents, :green_diagram; is_bold = false)
            dims = config_dimensions(cache)
            n_total = total_diagrams(cache)

            if rank == 0
                println("Evaluating ", dims, " = ", n_total, " Green's function diagrams at order ", order)
            end

            # Store diagram info: (n_topologies, n_directions, n_flavors_combo, n_exponents_combo, total)
            push!(solver_data.calc.green_diagram_info, (dims..., n_total))

            eval_green_given_order!(solver_data, cache, solver_params.green.is_bold, comm)

            push!(solver_data.calc.green_numerator_by_order,
                copy(solver_data.calc.green.vals))
        end

        # Compute Green's function by order (numerator / Z)
        for green_order in 1:(solver_params.green.expansion_order)
            propagator_index = green_order
            green_index = green_order
            green_at_order_data = solver_data.calc.green_numerator_by_order[green_index] ./
                                  solver_data.calc.partition_function_by_order[propagator_index]
            push!(solver_data.calc.green_by_order, green_at_order_data)
        end

        # Update final Green's function
        copyto!(solver_data.calc.green.vals, solver_data.calc.green_by_order[end])
        solver_data.calc.green.coeffs = QMBToolkitDLR.tau2dlr(
            solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.τ; axis = 3)
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

# MutableBackBone version - same logic, different type
function compute_diagram_prefactor(diagram::MutableBackBone, K0_inverse::Vector{Float64})
    prefactor = diagram.sign
    for (idx, Kexp) in enumerate(diagram.prefactor_Kexps)
        if Kexp != 0
            prefactor *= K0_inverse[idx]^Kexp
        end
    end
    return prefactor
end
