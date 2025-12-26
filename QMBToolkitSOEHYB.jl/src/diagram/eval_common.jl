### Evaluate the backbone diagrams to obtain their contributions
using MPI
using LinearAlgebra
using QMBToolkitDLR

"""
    eval_diagrams!(solver_data, solver_params, diagram_type, diagrams, comm)

Evaluate all diagrams for a given diagram type and accumulate results into solver_data.
Format-agnostic implementation using interface functions.

# Arguments
- `solver_data::SolverData{T, Format}`: The solver data containing model information and calculation results
- `solver_params::SolverParams`: Solver parameters including expansion orders and boldness settings
- `diagram_type::Symbol`: Type of diagrams to evaluate (`:propagator_diagram`, `:green_diagram`, or `:selfenergy_diagram`)
- `diagrams::Vector{Array{BackBone,4}}`: Vector of diagram arrays organized by expansion order
- `comm::MPI.Comm`: MPI communicator for parallel execution

For propagator diagrams, updates propagator and partition_function.
For Green's function diagrams, updates green.
"""
function eval_diagrams!(solver_data::SolverData{T, Format}, solver_params::SolverParams, diagram_type::Symbol, diagrams::Vector{Array{BackBone,4}}, comm::MPI.Comm) where {T, Format<:HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    propagator = get_propagator(solver_data.calc)  # Cache accessor result

    if diagram_type == :propagator_diagram &&
           !solver_params.propagator.is_bold
        # Clear partition function by order tracking
        empty!(solver_data.calc.partition_function_by_order)

        for order in 0:solver_params.propagator.expansion_order
            if rank == 0
                println("Evaluating propagator diagrams at order ", order)
            end
            eval_propagator_given_order!(solver_data, diagrams[order+1], comm)

            # Calculate and store partition function after this order
            # extract_at_tau is format-agnostic (returns dense matrix for both formats)
            propagator_at_beta = extract_at_tau(propagator.vals, solver_data.model.beta, solver_data.model.dlr_grid)
            Z_current = tr(propagator_at_beta)
            push!(solver_data.calc.partition_function_by_order, Z_current)
        end

        # Update final dlr coefficients (format-agnostic)
        propagator.coeffs = tau2dlr(solver_data.model.dlr_grid, propagator.vals, solver_data.model.dlr_grid.tau)
    elseif diagram_type == :green_diagram
        empty!(solver_data.calc.green_numerator_by_order)
        empty!(solver_data.calc.green_by_order)
        for order in 1:solver_params.green.expansion_order
            if rank == 0
                println("Evaluating Green's function diagrams at order ", order)
            end
            eval_green_given_order!(solver_data, diagrams[order], solver_params.green.is_bold, comm)

            # Store Green's function numerator after this order
            # Green's function is always stored as DensePropagator
            push!(solver_data.calc.green_numerator_by_order, DensePropagator{T}(copy(solver_data.calc.green.vals.data)))
        end

        # Compute Green's function by order (numerator / Z)
        # Green's function order i uses partition function from propagator order i-1
        for order in 1:solver_params.green.expansion_order
            propagator_order = order  # Green order 1 uses Z[1] (propagator order 0), etc.
            green_at_order_data = solver_data.calc.green_numerator_by_order[order].data ./ solver_data.calc.partition_function_by_order[propagator_order]
            push!(solver_data.calc.green_by_order, DensePropagator{T}(green_at_order_data))
        end

        # Update final Green's function
        copyto!(solver_data.calc.green.vals, solver_data.calc.green_by_order[end])
        # Green's function coefficients use dense DLR transform
        solver_data.calc.green.coeffs = tau2dlr_dense(solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.tau)
    else
        error("Unknown diagram type: $diagram_type. Supported types are: :propagator_diagram, :green_diagram, :selfenergy_diagram")
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
