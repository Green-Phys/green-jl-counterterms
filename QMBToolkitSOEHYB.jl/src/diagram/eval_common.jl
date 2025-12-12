### Evaluate the backbone diagrams to obtain their contributions
using MPI
using LinearAlgebra
using QMBToolkitDLR

"""
    eval_diagrams!(solver_data, solver_params, diagram_type, diagrams, comm)

Evaluate all diagrams for a given diagram type and accumulate results into solver_data.

# Arguments
- `solver_data::SolverData`: The solver data containing model information and calculation results
- `solver_params::SolverParams`: Solver parameters including expansion orders and boldness settings
- `diagram_type::Symbol`: Type of diagrams to evaluate (`:propagator_diagram`, `:green_diagram`, or `:selfenergy_diagram`)
- `diagrams::Vector{Array{BackBone,4}}`: Vector of diagram arrays organized by expansion order
- `comm::MPI.Comm`: MPI communicator for parallel execution

For propagator diagrams, updates `solver_data.calc.propagator` and `solver_data.calc.partition_function`.
For Green's function diagrams, updates `solver_data.calc.green`.
"""
function eval_diagrams!(solver_data::SolverData, solver_params::SolverParams, diagram_type::Symbol, diagrams::Vector{Array{BackBone,4}}, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
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
            propagator_coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.propagator.vals, solver_data.model.dlr_grid.tau)
            propagator_at_beta = dlr2tau(solver_data.model.dlr_grid, propagator_coeffs, [solver_data.model.beta])[1, :, :]
            Z_current = tr(propagator_at_beta)
            push!(solver_data.calc.partition_function_by_order, Z_current)
        end

        # Update final dlr coefficients
        solver_data.calc.propagator.coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.propagator.vals, solver_data.model.dlr_grid.tau)
    elseif diagram_type == :green_diagram
        for order in 1:solver_params.green.expansion_order
            if rank == 0
                println("Evaluating Green's function diagrams at order ", order)
            end
            eval_green_given_order!(solver_data, diagrams[order], solver_params.green.is_bold, comm)

            # Store Green's function numerator after this order
            push!(solver_data.calc.green_numerator_by_order, copy(solver_data.calc.green.vals))
        end

        # Compute Green's function by order (numerator / Z)
        # Green's function order i uses partition function from propagator order i-1
        for order in 1:solver_params.green.expansion_order
            propagator_order = order  # Green order 1 uses Z[1] (propagator order 0), etc.
            green_at_order = solver_data.calc.green_numerator_by_order[order] ./ solver_data.calc.partition_function_by_order[propagator_order]
            push!(solver_data.calc.green_by_order, green_at_order)
        end

        # Update final Green's function (using last order)
        solver_data.calc.green.vals .= solver_data.calc.green_by_order[end]

        # update dlr coefficients
        solver_data.calc.green.coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.tau)
    else
        error("Unknown diagram type: $diagram_type. Supported types are: :propagator_diagram, :green_diagram, :selfenergy_diagram")
    end
end

# All operator multiplication, hybridization, and convolution functions
# have been moved to src/dense/ folder for better organization
