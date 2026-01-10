"""
Unified backbone diagram evaluation.

Format-agnostic implementation of backbone diagram evaluation that works with both
Dense and BlockDense representations through interface functions.

# Key Functions
- `get_operator(node, model_data)`: Get operator for a diagram node
- `multiplication_convolution(...)`: Forward backbone evaluation (right to left)
- `multiplication_convolution_reverse(...)`: Reverse backbone evaluation (left to right)
- `eval_propagator_forward_backbone(...)`: Propagator diagram evaluation
- `eval_green_forward_backbone(...)`: Forward Green's function backbone
- `eval_green_reverse_backbone(...)`: Reverse Green's function backbone

# Interface Functions Used
- get_flux, compute_new_flux, compute_new_flux_reverse: Flux handling
- get_result, get_hyb_temp, get_op_product, etc.: Buffer accessors
- multiply_op_prop!, multiply_prop_op!: Operator multiplications
- apply_hybridization_inplace!: Hybridization application
- tau2dlr!, do_convolution_dlr2tau!: DLR operations
"""

using QMBToolkitDLR

#=============================================================================
# Operator Lookup
=============================================================================#

"""
    get_operator(node::BackBoneNode, model_data::ModelData)

Get the operator for a given node from model data.

This is a format-agnostic function that uses interface accessors to retrieve
the appropriate operator (Dense or BlockDense) based on the model_data type.

# Arguments
- `node::BackBoneNode`: Node containing operator type and indices
- `model_data::ModelData`: Model data with operators

# Returns
Appropriate operator based on node type:
- `:cdag` with bar_index=-1: Creation operator from exact diagonalization
- `:c` with bar_index=-1: Annihilation operator from exact diagonalization
- `:cdag` with bar_index≠-1: SOE-transformed creation operator
- `:c` with bar_index≠-1: SOE-transformed annihilation operator
"""
function get_operator(node::BackBoneNode, model_data::ModelData)
    if node.opt_type == :cdag
        if node.bar_index == -1
            return get_cre_opts(model_data)[node.flavor]
        else
            return get_cre_opts_bar(model_data)[node.flavor, node.bar_index]
        end
    else  # node.opt_type == :c
        if node.bar_index == -1
            return get_ann_opts(model_data)[node.flavor]
        else
            return get_ann_opts_bar(model_data)[node.flavor, node.bar_index]
        end
    end
end

#=============================================================================
# Core Backbone Convolution Operations
=============================================================================#

"""
    multiplication_convolution(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from right to left.

# Algorithm
Evaluates diagram backbone from right to left:
```
|Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
```

Starting from the rightmost edge, accumulates:
1. Apply hybridizations attached to edge
2. Multiply with next node operator
3. Apply hybridization attached to node
4. Convolve with next edge's propagator (with its hybridizations)

# Arguments
- `nodes::Vector{BackBoneNode}`: Diagram nodes (length M)
- `edges::Vector{BackBoneEdge}`: Diagram edges (length M-1)
- `dlr_grid::DLRGrid`: DLR grid for τ-space operations
- `model_data::ModelData`: Model data with operators and hybridization
- `propagator`: Propagator on DLR τ-grid (DensePropagator or BlockDensePropagator)
- `buffers`: Pre-allocated work buffers (DenseBackboneBuffers or BlockDenseBackboneBuffers)

# Returns
Reference to the result buffer containing diagram contribution on DLR τ-grid.

# Notes
Pre-allocated buffers are reused across all diagram evaluations for efficiency.
All operations are in-place to minimize allocations.

# Buffer Usage (7 roles)
- `result`: Accumulates backbone result
- `hyb_temp`: Scratch for hybridization on result
- `op_product`: Holds operator multiplication output
- `conv_operand`: Propagator copy for convolution
- `conv_hyb_temp`: Scratch for hybridization on conv operand
- `conv_dlr`: DLR transform of conv operand
- `result_dlr`: DLR transform of result
"""
function multiplication_convolution(
        nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid,
        model_data::ModelData{T}, propagator, buffers) where {T <: Number}
    length(nodes) == length(edges) + 1 ||
        error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get initial flux (nothing for Dense, OperatorFlux for BlockDense)
    current_flux = get_flux(propagator)

    # Get result buffer for current flux and copy propagator into it
    result = get_result(buffers, current_flux)
    copyto!(result, propagator)

    # Get hyb_temp buffer for current flux
    hyb_temp = get_hyb_temp(buffers, current_flux)

    # Process edges from right to left
    for i in 1:(length(edges) - 1)
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(hyb_temp, result, hyb, model_data, is_reverse = false)
            end
        end

        # Multiply with the next node operator
        node = nodes[i + 1]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)

            # Compute new flux after operator multiplication
            new_flux = compute_new_flux(current_flux, op)

            # Get op_product buffer for the new flux
            op_product = get_op_product(buffers, new_flux)
            fill!(op_product, zero(T))

            # Perform multiplication: op * result -> op_product
            multiply_op_prop!(op_product, op, result)

            # Update current flux and get new result buffer
            current_flux = new_flux
            result = get_result(buffers, current_flux)
            copyto!(result, op_product)

            # Update hyb_temp to match new flux
            hyb_temp = get_hyb_temp(buffers, current_flux)
        end

        # Apply hybridization attached to the node
        apply_hybridization_inplace!(
            hyb_temp, result, node.attached_hyb, model_data, is_reverse = false)

        # Calculate the convolution with the next edge
        edge_plus1 = edges[i + 1]

        # Get conv_operand buffer (fixed zero flux) and copy propagator
        conv_operand = get_conv_operand(buffers)
        copyto!(conv_operand, propagator)

        # Get conv_hyb_temp buffer (fixed zero flux)
        conv_hyb_temp = get_conv_hyb_temp(buffers)

        for hyb in edge_plus1.attached_hybs
            apply_hybridization_inplace!(
                conv_hyb_temp, conv_operand, hyb, model_data, is_reverse = false)
        end

        # Transform to DLR using buffers
        conv_dlr = get_conv_dlr(buffers)
        tau2dlr!(conv_dlr, dlr_grid, conv_operand, dlr_grid.tau)

        result_dlr = get_result_dlr(buffers, current_flux)
        tau2dlr!(result_dlr, dlr_grid, result, dlr_grid.tau)

        # Perform convolution - result is written back to result buffer
        do_convolution_dlr2tau!(result, model_data.convolution_workspace, dlr_grid,
            model_data.kernel_cache, conv_dlr, result_dlr)
    end

    return result
end

"""
    multiplication_convolution_reverse(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from left to right.

Similar to `multiplication_convolution` but traverses in reverse direction with swapped K± hybridizations.

# Key Differences from Forward Direction
- Traverses from left to right (indices M down to 1)
- Uses `is_reverse=true` for node-attached hybridizations (swaps K+ ↔ K-)
- Uses `multiply_prop_op!` instead of `multiply_op_prop!` for operator multiplication
- Convolution operand order is reversed: result first, then conv_operand

# Arguments
Same as `multiplication_convolution`.

# Returns
Reference to the result buffer containing diagram contribution on DLR τ-grid.
"""
function multiplication_convolution_reverse(
        nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid,
        model_data::ModelData{T}, propagator, buffers) where {T <: Number}
    length(nodes) == length(edges) + 1 ||
        error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get initial flux (nothing for Dense, OperatorFlux for BlockDense)
    current_flux = get_flux(propagator)

    # Get result buffer for current flux and copy propagator into it
    result = get_result(buffers, current_flux)
    copyto!(result, propagator)

    # Get hyb_temp buffer for current flux
    hyb_temp = get_hyb_temp(buffers, current_flux)

    # Process edges from left to right
    for i in length(edges):-1:2
        edge = edges[i]
        if i == length(edges)
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(hyb_temp, result, hyb, model_data, is_reverse = false)
            end
        end

        node = nodes[i]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)

            # Compute new flux after operator multiplication (reverse direction)
            new_flux = compute_new_flux_reverse(current_flux, op)

            # Get op_product buffer for the new flux
            op_product = get_op_product(buffers, new_flux)
            fill!(op_product, zero(T))

            # Perform multiplication: result * op -> op_product
            multiply_prop_op!(op_product, result, op)

            # Update current flux and get new result buffer
            current_flux = new_flux
            result = get_result(buffers, current_flux)
            copyto!(result, op_product)

            # Update hyb_temp to match new flux
            hyb_temp = get_hyb_temp(buffers, current_flux)
        end

        # Apply hybridization with is_reverse=true (swaps K+ ↔ K-)
        apply_hybridization_inplace!(
            hyb_temp, result, node.attached_hyb, model_data, is_reverse = true)

        # Calculate the convolution with the next edge
        edge_minus1 = edges[i - 1]

        # Get conv_operand buffer (fixed zero flux) and copy propagator
        conv_operand = get_conv_operand(buffers)
        copyto!(conv_operand, propagator)

        # Get conv_hyb_temp buffer (fixed zero flux)
        conv_hyb_temp = get_conv_hyb_temp(buffers)

        for hyb in edge_minus1.attached_hybs
            apply_hybridization_inplace!(
                conv_hyb_temp, conv_operand, hyb, model_data, is_reverse = false)
        end

        # Transform to DLR using buffers (note: order reversed)
        result_dlr = get_result_dlr(buffers, current_flux)
        tau2dlr!(result_dlr, dlr_grid, result, dlr_grid.tau)

        conv_dlr = get_conv_dlr(buffers)
        tau2dlr!(conv_dlr, dlr_grid, conv_operand, dlr_grid.tau)

        # Perform convolution (reversed order) - result is written back to result buffer
        do_convolution_dlr2tau!(result, model_data.convolution_workspace, dlr_grid,
            model_data.kernel_cache, result_dlr, conv_dlr)
    end

    return result
end

#=============================================================================
# Propagator Backbone Evaluation
=============================================================================#

"""
    eval_propagator_forward_backbone(diagram, solver_data, buffers)

Evaluate a single propagator diagram using the unified backbone evaluation.

This is the format-agnostic implementation that works for both Dense and BlockDense modes.
Called by `eval_propagator_given_order!`.

# Arguments
- `diagram::BackBone`: The diagram to evaluate
- `solver_data::SolverData{T}`: Solver data containing model information
- `buffers`: Pre-allocated work buffers (format-specific)

# Returns
Diagram contribution in imaginary time (format-specific propagator type).
"""
function eval_propagator_forward_backbone(diagram::BackBone, solver_data::SolverData{T}, buffers) where {T <:
                                                                                                         Number}
    bare_propagator = get_bare_propagator(solver_data.model)
    return multiplication_convolution(
        diagram.nodes, diagram.edges,
        solver_data.model.dlr_grid, solver_data.model,
        bare_propagator.vals,
        buffers,
    )
end

#=============================================================================
# Green's Function Backbone Evaluation
=============================================================================#

"""
    eval_green_forward_backbone(diagram, split_point, solver_data, propagator, buffers)

Evaluate forward backbone contribution from node 1 to split_point.
Format-agnostic: works with both Dense and BlockDense propagators.

# Arguments
- `diagram::BackBone`: The diagram being evaluated
- `split_point::Int`: Position of the external annihilation operator
- `solver_data::SolverData{T}`: Solver data containing model information
- `propagator`: Propagator to use (DensePropagator or BlockDensePropagator)
- `buffers`: Evaluation buffers containing forward convolution buffers

# Returns
Forward backbone contribution on DLR τ-grid.
"""
function eval_green_forward_backbone(diagram::BackBone, split_point::Int,
        solver_data::SolverData{T}, propagator, buffers) where {T}
    return multiplication_convolution(
        diagram.nodes[1:split_point],
        diagram.edges[1:(split_point - 1)],
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator,
        buffers.convolution_buffers.fwd_buffers,
    )
end

"""
    eval_green_reverse_backbone(diagram, split_point, solver_data, propagator, buffers)

Evaluate reverse backbone contribution from split_point to end.
Format-agnostic: works with both Dense and BlockDense propagators.

# Arguments
- `diagram::BackBone`: The diagram being evaluated
- `split_point::Int`: Position of the external annihilation operator
- `solver_data::SolverData{T}`: Solver data containing model information
- `propagator`: Propagator to use (DensePropagator or BlockDensePropagator)
- `buffers`: Evaluation buffers containing reverse convolution buffers

# Returns
Reverse backbone contribution on DLR τ-grid.
"""
function eval_green_reverse_backbone(diagram::BackBone, split_point::Int,
        solver_data::SolverData{T}, propagator, buffers) where {T}
    return multiplication_convolution_reverse(
        diagram.nodes[split_point:end],
        diagram.edges[split_point:end],
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator,
        buffers.convolution_buffers.rev_buffers,
    )
end
