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

"""
    get_operator_matrix(node::BackBoneNode, model_data::ModelData)

Get the operator matrix for a given node from model data.
"""
function get_operator_matrix(node::BackBoneNode, model_data::ModelData)
    if node.opt_type == :cdag && node.bar_index == -1
        return model_data.cre_opts[node.flavor]
    elseif node.opt_type == :c && node.bar_index == -1
        return model_data.ann_opts[node.flavor]
    elseif node.opt_type == :cdag && node.bar_index != -1
        return model_data.cre_opts_SOE[node.flavor, node.bar_index]
    elseif node.opt_type == :c && node.bar_index != -1
        return model_data.ann_opts_SOE[node.flavor, node.bar_index]
    end
end

"""
    apply_hybridization_inplace!(temp_buffer, target, hyb, model_data; is_reverse=false)

Apply hybridization function to target array in-place, updating target with the result.
Uses pre-allocated temp_buffer to avoid allocations.

When `is_reverse=true`, swaps KPlus ↔ KMinus (used in reverse convolution).
"""
function apply_hybridization_inplace!(temp_buffer::Array{Float64,3}, target::Array{Float64,3}, hyb::HybridizationState, model_data::ModelData; is_reverse::Bool=false)
    if !is_nohyb(hyb)
        # Select the appropriate hybridization array (swap if reverse)
        if (is_kplus(hyb) && !is_reverse) || (is_kminus(hyb) && is_reverse)
            # Use KPlus
            l = is_kplus(hyb) ? get_kplus_index(hyb) : get_kminus_index(hyb)
            vec = @view(model_data.hyb.KPlus_on_DLRtau[:, l])
        else
            # Use KMinus
            l = is_kminus(hyb) ? get_kminus_index(hyb) : get_kplus_index(hyb)
            vec = @view(model_data.hyb.KMinus_on_DLRtau[:, l])
        end
        _multiplication_13!(temp_buffer, vec, target)
        copyto!(target, temp_buffer)
    end
    # For NoHyb, do nothing (target unchanged)
end

"""
    multiplication_convolution(nodes, edges, dlr_grid, model_data, propagator, temp_buffer, result_buffer, convolution_buffer)

Calculate backbone diagram contribution via multiplication and convolution from right to left.

Pre-allocated buffers are reused across all diagram evaluations for efficiency:
- `temp_buffer`: Temporary storage for in-place matrix operations
- `result_buffer`: Working buffer for accumulated result
- `convolution_buffer`: Working buffer for convolution operands
"""
function multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData, propagator::DLRData, temp_buffer::Array{Float64,3}, result_buffer::Array{Float64,3}, convolution_buffer::Array{Float64,3})
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")
    # calculate the multiplication and convolution from right to left ignoring the two boundary nodes
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result_buffer, propagator.vals)
    result = result_buffer
    for i in 1:length(edges)-1
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(temp_buffer, result, hyb, model_data, is_reverse=false)
            end
        end
        # multiply with the next node operator
        node = nodes[i+1]
        if node.opt_type !== :empty
            op_matrix = get_operator_matrix(node, model_data)
            _multiplication_23!(temp_buffer, op_matrix, result)
            copyto!(result, temp_buffer)
        end

        # Apply hybridization attached to the node
        apply_hybridization_inplace!(temp_buffer, result, node.attached_hyb, model_data, is_reverse=false)

        # calculate the convolution with the next edge
        edge_plus1 = edges[i+1]
        copyto!(convolution_buffer, propagator.vals)
        to_be_convoluted = convolution_buffer
        for hyb in edge_plus1.attached_hybs
            apply_hybridization_inplace!(temp_buffer, to_be_convoluted, hyb, model_data, is_reverse=false)
        end

        result = do_convolution_dlr2tau(dlr_grid, tau2dlr(dlr_grid, to_be_convoluted, dlr_grid.tau), tau2dlr(dlr_grid, result, dlr_grid.tau))
    end
    return result
end


"""
    multiplication_convolution_reverse(nodes, edges, dlr_grid, model_data, propagator, temp_buffer, result_buffer, convolution_buffer)

Calculate backbone diagram contribution via multiplication and convolution from left to right.
Similar to `multiplication_convolution` but traverses in reverse direction with swapped K± hybridizations.
"""
function multiplication_convolution_reverse(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData, propagator::DLRData, temp_buffer::Array{Float64,3}, result_buffer::Array{Float64,3}, convolution_buffer::Array{Float64,3})
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")
    # calculate the multiplication and convolution from left to right ignoring the two boundary nodes (the matrix multiplications is done in the original order)
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result_buffer, propagator.vals)
    result = result_buffer
    for i in length(edges):-1:2
        edge = edges[i]
        if i == length(edges)
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(temp_buffer, result, hyb, model_data, is_reverse=false)
            end
        end
        node = nodes[i]
        if node.opt_type !== :empty
            op_matrix = get_operator_matrix(node, model_data)
            _multiplication_32!(temp_buffer, result, op_matrix)
            copyto!(result, temp_buffer)
        end

        # Apply hybridization with is_reverse=true (swaps K+ ↔ K-)
        apply_hybridization_inplace!(temp_buffer, result, node.attached_hyb, model_data, is_reverse=true)

        # calculate the convolution with the next edge
        edge_minus1 = edges[i-1]
        copyto!(convolution_buffer, propagator.vals)
        to_be_convoluted = convolution_buffer
        for hyb in edge_minus1.attached_hybs
            apply_hybridization_inplace!(temp_buffer, to_be_convoluted, hyb, model_data, is_reverse=false)
        end

        result = do_convolution_dlr2tau(dlr_grid, tau2dlr(dlr_grid, result, dlr_grid.tau), tau2dlr(dlr_grid, to_be_convoluted, dlr_grid.tau))
    end
    return result
end


# In-place version: writes result to temp_buffer, then caller copies back to input
function _multiplication_23!(temp_buffer::Array{Float64,3}, A::Array{Float64,2}, B::Array{Float64,3})
    n_tau = size(B, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer, k, :, :), A, view(B, k, :, :))
    end
    return temp_buffer
end

# Allocating version for backward compatibility
function _multiplication_23(A::Array{Float64,2}, B::Array{Float64,3})
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A * B[k, :, :]
    end
    return C
end

function _multiplication_32!(temp_buffer::Array{Float64,3}, A::Array{Float64,3}, B::Array{Float64,2})
    n_tau = size(A, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer, k, :, :), view(A, k, :, :), B)
    end
    return temp_buffer
end

function _multiplication_32(A::Array{Float64,3}, B::Array{Float64,2})
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k, :, :] * B
    end
    return C
end

function _multiplication_13!(temp_buffer::Array{Float64,3}, A::AbstractVector{Float64}, B::Array{Float64,3})
    n_tau = size(B, 1)
    @inbounds for k in 1:n_tau
        @views temp_buffer[k, :, :] .= A[k] .* B[k, :, :]
    end
    return temp_buffer
end

function _multiplication_13(A::Array{Float64,1}, B::Array{Float64,3})
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k] .* B[k, :, :]
    end
    return C
end

function _multiplication_33!(temp_buffer::Array{Float64,3}, A::Array{Float64,3}, B::Array{Float64,3})
    n_tau = size(A, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer, k, :, :), view(A, k, :, :), view(B, k, :, :))
    end
    return temp_buffer
end

function _multiplication_33(A::Array{Float64,3}, B::Array{Float64,3})
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k, :, :] * B[k, :, :]
    end
    return C
end


function _trace_3(A::Array{Float64,3})
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    o1 == o2 || error("Matrix to be traced must be square: got dimensions $o1 x $o2")
    trace_vals = zeros(Float64, n_tau)
    for k in 1:n_tau
        trace_vals[k] = tr(A[k, :, :])
    end
    return trace_vals
end

# In-place version: writes result to trace_buffer
function _trace_3!(trace_buffer::Vector{Float64}, A::Array{Float64,3})
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    o1 == o2 || error("Matrix to be traced must be square: got dimensions $o1 x $o2")
    @inbounds for k in 1:n_tau
        trace_buffer[k] = tr(A[k, :, :])
    end
    return trace_buffer
end


"""
    apply_hybridization(hyb::HybridizationState, result::Array{Float64,3}, model_data::ModelData)

Apply a hybridization function multiplication to the result array.

For KPlus hybridizations, multiplies by KPlus_on_DLRtau.
For KMinus hybridizations, multiplies by KMinus_on_DLRtau.
For NoHyb, returns the result unchanged.
"""
function apply_hybridization(hyb::HybridizationState, result::Array{Float64,3}, model_data::ModelData; is_reverse::Bool=false)
    if !is_nohyb(hyb)
        if (is_kplus(hyb) && !is_reverse) || (is_kminus(hyb) && is_reverse)
            # Use KPlus
            l = is_kplus(hyb) ? get_kplus_index(hyb) : get_kminus_index(hyb)
            return _multiplication_13(model_data.hyb.KPlus_on_DLRtau[:, l], result)
        else
            # Use KMinus
            l = is_kminus(hyb) ? get_kminus_index(hyb) : get_kplus_index(hyb)
            return _multiplication_13(model_data.hyb.KMinus_on_DLRtau[:, l], result)
        end
    else
        return result
    end
end
