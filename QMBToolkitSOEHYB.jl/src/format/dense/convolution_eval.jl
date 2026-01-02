"""
Backbone diagram evaluation for dense representation.

Functions for evaluating diagram contributions through multiplication and convolution operations.
These implement the core diagram evaluation logic for propagator and Green's function expansions.

Note: get_operator is defined in diagram/backbone_common.jl as a format-agnostic function.
"""

using QMBToolkitDLR
using QMBToolkitED.Dense: DenseOperator

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
- `model_data::ModelData{T}`: Model data with operators and hybridization
- `propagator::DensePropagator{T}`: Propagator on DLR τ-grid
- `buffers::DenseBackboneBuffers{T}`: Pre-allocated work buffers

# Returns
- `DensePropagator{T}`: Diagram contribution on DLR τ-grid (returns reference to buffers.result)

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
function multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T}, propagator::DensePropagator{T}, buffers::DenseBackboneBuffers{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Extract buffers with clear role names
    result = buffers.result
    hyb_temp = buffers.hyb_temp
    op_product = buffers.op_product
    conv_operand = buffers.conv_operand
    conv_hyb_temp = buffers.conv_hyb_temp
    conv_dlr = buffers.conv_dlr
    result_dlr = buffers.result_dlr

    # calculate the multiplication and convolution from right to left ignoring the two boundary nodes
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result, propagator)

    for i in 1:length(edges)-1
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(hyb_temp, result, hyb, model_data, is_reverse=false)
            end
        end
        # multiply with the next node operator
        node = nodes[i+1]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)
            _multiplication_23!(op_product, op, result)
            copyto!(result, op_product)
        end

        # Apply hybridization attached to the node
        apply_hybridization_inplace!(hyb_temp, result, node.attached_hyb, model_data, is_reverse=false)

        # calculate the convolution with the next edge
        edge_plus1 = edges[i+1]
        copyto!(conv_operand, propagator)
        for hyb in edge_plus1.attached_hybs
            apply_hybridization_inplace!(conv_hyb_temp, conv_operand, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR (in-place) and perform convolution
        tau2dlr_dense!(conv_dlr, dlr_grid, conv_operand, dlr_grid.tau)
        tau2dlr_dense!(result_dlr, dlr_grid, result, dlr_grid.tau)
        do_convolution_dlr2tau_dense!(result, model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, conv_dlr, result_dlr)
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
- Uses `_multiplication_32!` instead of `_multiplication_23!` for operator multiplication
- Convolution operand order is reversed: result first, then to_be_convoluted

# Arguments
Same as `multiplication_convolution`.

# Returns
- `DensePropagator{T}`: Diagram contribution on DLR τ-grid (returns reference to buffers.result)
"""
function multiplication_convolution_reverse(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T}, propagator::DensePropagator{T}, buffers::DenseBackboneBuffers{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Extract buffers with clear role names
    result = buffers.result
    hyb_temp = buffers.hyb_temp
    op_product = buffers.op_product
    conv_operand = buffers.conv_operand
    conv_hyb_temp = buffers.conv_hyb_temp
    conv_dlr = buffers.conv_dlr
    result_dlr = buffers.result_dlr

    # calculate the multiplication and convolution from left to right ignoring the two boundary nodes (the matrix multiplications is done in the original order)
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result, propagator)

    for i in length(edges):-1:2
        edge = edges[i]
        if i == length(edges)
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(hyb_temp, result, hyb, model_data, is_reverse=false)
            end
        end
        node = nodes[i]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)
            _multiplication_32!(op_product, result, op)
            copyto!(result, op_product)
        end

        # Apply hybridization with is_reverse=true (swaps K+ ↔ K-)
        apply_hybridization_inplace!(hyb_temp, result, node.attached_hyb, model_data, is_reverse=true)

        # calculate the convolution with the next edge
        edge_minus1 = edges[i-1]
        copyto!(conv_operand, propagator)
        for hyb in edge_minus1.attached_hybs
            apply_hybridization_inplace!(conv_hyb_temp, conv_operand, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR (in-place) and perform convolution
        # Note: order is reversed compared to forward direction
        tau2dlr_dense!(result_dlr, dlr_grid, result, dlr_grid.tau)
        tau2dlr_dense!(conv_dlr, dlr_grid, conv_operand, dlr_grid.tau)
        do_convolution_dlr2tau_dense!(result, model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, result_dlr, conv_dlr)
    end

    return result
end

