### Evaluate the backbone diagrams to obtain their contributions
using MPI
using LinearAlgebra
using QMBToolkitDLR

# Note: Accessor functions get_propagator() and get_selfenergy() are defined in
# src/solver/Data.jl and dispatch via format_data field

"""
    extract_at_tau_blockdense(propagator::BlockDensePropagator{T}, tau::Float64, dlr_grid::DLRGrid)

Extract propagator value at specific tau by DLR interpolation, return as dense matrix.
"""
function extract_at_tau_blockdense(propagator::BlockDensePropagator{T}, tau::Float64, dlr_grid::DLRGrid) where T
    # Convert to DLR coefficients
    prop_dlr = tau2dlr_blockdense(dlr_grid, propagator, dlr_grid.tau)

    # Evaluate at the requested tau
    prop_at_tau = dlr2tau_blockdense(dlr_grid, prop_dlr, [tau])

    # Convert single-time-slice BlockDensePropagator to dense matrix
    return blockdense_to_dense_single_tau(prop_at_tau)
end

"""
    blockdense_to_dense_single_tau(bda::BlockDensePropagator{T})

Convert a BlockDensePropagator with single time slice (n_tau=1) to a dense matrix.
"""
function blockdense_to_dense_single_tau(bda::BlockDensePropagator{T}) where T
    # Determine total dimension from basis
    n_total = sum(sector.dim for sector in bda.basis.sectors)
    result = zeros(T, n_total, n_total)

    # Fill in blocks
    for (key, block) in bda.blocks
        (i, j) = key
        dim_i = bda.basis.sectors[i].dim
        dim_j = bda.basis.sectors[j].dim

        # Get row and column offsets
        row_offset = sum(bda.basis.sectors[k].dim for k in 1:i-1; init=0)
        col_offset = sum(bda.basis.sectors[k].dim for k in 1:j-1; init=0)

        # Extract single time slice (should be at index 1)
        @assert size(block, 1) == 1 "Expected single time slice"
        result[row_offset+1:row_offset+dim_i, col_offset+1:col_offset+dim_j] = block[1, :, :]
    end

    return result
end

"""
    eval_diagrams!(solver_data, solver_params, diagram_type, diagrams, comm)

Evaluate all diagrams for a given diagram type and accumulate results into solver_data.

# Arguments
- `solver_data::SolverData`: The solver data containing model information and calculation results
- `solver_params::SolverParams`: Solver parameters including expansion orders and boldness settings
- `diagram_type::Symbol`: Type of diagrams to evaluate (`:propagator_diagram`, `:green_diagram`, or `:selfenergy_diagram`)
- `diagrams::Vector{Array{BackBone,4}}`: Vector of diagram arrays organized by expansion order
- `comm::MPI.Comm`: MPI communicator for parallel execution

For propagator diagrams, updates propagator and partition_function.
For Green's function diagrams, updates green.
"""
function eval_diagrams!(solver_data::SolverData{T, DenseFormat}, solver_params::SolverParams, diagram_type::Symbol, diagrams::Vector{Array{BackBone,4}}, comm::MPI.Comm) where T
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
            propagator_dlr = tau2dlr_dense(solver_data.model.dlr_grid, propagator.vals, solver_data.model.dlr_grid.tau)
            propagator_at_beta = dlr2tau_dense(solver_data.model.dlr_grid, propagator_dlr, [solver_data.model.beta])
            Z_current = tr(propagator_at_beta.data[1, :, :])
            push!(solver_data.calc.partition_function_by_order, Z_current)
        end

        # Update final dlr coefficients
        propagator.coeffs = tau2dlr_dense(solver_data.model.dlr_grid, propagator.vals, solver_data.model.dlr_grid.tau)
    elseif diagram_type == :green_diagram
        for order in 1:solver_params.green.expansion_order
            if rank == 0
                println("Evaluating Green's function diagrams at order ", order)
            end
            eval_green_given_order!(solver_data, diagrams[order], solver_params.green.is_bold, comm)

            # Store Green's function numerator after this order
            push!(solver_data.calc.green_numerator_by_order, DensePropagator{T}(copy(solver_data.calc.green.vals.data)))
        end

        # Compute Green's function by order (numerator / Z)
        # Green's function order i uses partition function from propagator order i-1
        for order in 1:solver_params.green.expansion_order
            propagator_order = order  # Green order 1 uses Z[1] (propagator order 0), etc.
            green_at_order_data = solver_data.calc.green_numerator_by_order[order].data ./ solver_data.calc.partition_function_by_order[propagator_order]
            push!(solver_data.calc.green_by_order, DensePropagator{T}(green_at_order_data))
        end

        # Update final Green's function (using last order)
        copyto!(solver_data.calc.green.vals, solver_data.calc.green_by_order[end])

        # update dlr coefficients
        solver_data.calc.green.coeffs = tau2dlr_dense(solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.tau)
    else
        error("Unknown diagram type: $diagram_type. Supported types are: :propagator_diagram, :green_diagram, :selfenergy_diagram")
    end
end

# BlockDenseFormat version
function eval_diagrams!(solver_data::SolverData{T, BlockDenseFormat}, solver_params::SolverParams, diagram_type::Symbol, diagrams::Vector{Array{BackBone,4}}, comm::MPI.Comm) where T
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
            # For BlockDense, convert to dense at β for trace calculation
            propagator_at_beta_dense = extract_at_tau_blockdense(propagator.vals, solver_data.model.beta, solver_data.model.dlr_grid)
            Z_current = tr(propagator_at_beta_dense)
            push!(solver_data.calc.partition_function_by_order, Z_current)
        end

        # Update final dlr coefficients
        propagator.coeffs = tau2dlr_blockdense(solver_data.model.dlr_grid, propagator.vals, solver_data.model.dlr_grid.tau)
    elseif diagram_type == :green_diagram
        for order in 1:solver_params.green.expansion_order
            if rank == 0
                println("Evaluating Green's function diagrams at order ", order)
            end
            eval_green_given_order!(solver_data, diagrams[order], solver_params.green.is_bold, comm)

            # Store Green's function numerator after this order
            push!(solver_data.calc.green_numerator_by_order, DensePropagator{T}(copy(solver_data.calc.green.vals.data)))
        end

        # Compute Green's function by order (numerator / Z)
        for order in 1:solver_params.green.expansion_order
            propagator_order = order
            green_at_order_data = solver_data.calc.green_numerator_by_order[order].data ./ solver_data.calc.partition_function_by_order[propagator_order]
            push!(solver_data.calc.green_by_order, DensePropagator{T}(green_at_order_data))
        end

        # Update final Green's function
        copyto!(solver_data.calc.green.vals, solver_data.calc.green_by_order[end])
        solver_data.calc.green.coeffs = tau2dlr_dense(solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.tau)
    else
        error("Unknown diagram type: $diagram_type. Supported types are: :propagator_diagram, :green_diagram, :selfenergy_diagram")
    end
end

# All operator multiplication, hybridization, and convolution functions
# have been moved to src/dense/ folder for better organization
