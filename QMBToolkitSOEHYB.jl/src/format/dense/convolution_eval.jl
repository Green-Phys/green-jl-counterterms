"""
Backbone diagram convolution evaluation for dense representation.

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
- `buffers::PropagatorConvolutionBuffers{T}`: Pre-allocated work buffers

# Returns
- `DensePropagator{T}`: Diagram contribution on DLR τ-grid (returns reference to buffers.backbone_accumulator)

# Notes
Pre-allocated buffers are reused across all diagram evaluations for efficiency.
All operations are in-place to minimize allocations.
"""
function multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T}, propagator::DensePropagator{T}, buffers::PropagatorConvolutionBuffers{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Extract buffers
    temp_buffer = buffers.operator_product_buffer
    result_buffer = buffers.backbone_accumulator
    convolution_buffer = buffers.convolution_operand_buffer
    dlr_buffer_1 = buffers.dlr_buffer_1
    dlr_buffer_2 = buffers.dlr_buffer_2

    # calculate the multiplication and convolution from right to left ignoring the two boundary nodes
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result_buffer, propagator)

    for i in 1:length(edges)-1
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(temp_buffer, result_buffer, hyb, model_data, is_reverse=false)
            end
        end
        # multiply with the next node operator
        node = nodes[i+1]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)
            _multiplication_23!(temp_buffer, op, result_buffer)
            copyto!(result_buffer, temp_buffer)
        end

        # Apply hybridization attached to the node
        apply_hybridization_inplace!(temp_buffer, result_buffer, node.attached_hyb, model_data, is_reverse=false)

        # calculate the convolution with the next edge
        edge_plus1 = edges[i+1]
        copyto!(convolution_buffer, propagator)
        for hyb in edge_plus1.attached_hybs
            apply_hybridization_inplace!(temp_buffer, convolution_buffer, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR (in-place) and perform convolution (result written to temp_buffer, then copied back)
        tau2dlr_dense!(dlr_buffer_1, dlr_grid, convolution_buffer, dlr_grid.tau)
        tau2dlr_dense!(dlr_buffer_2, dlr_grid, result_buffer, dlr_grid.tau)
        do_convolution_dlr2tau_dense!(result_buffer, model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, dlr_buffer_1, dlr_buffer_2)
    end

    return result_buffer
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
- `DensePropagator{T}`: Diagram contribution on DLR τ-grid (returns reference to buffers.backbone_accumulator)
"""
function multiplication_convolution_reverse(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T}, propagator::DensePropagator{T}, buffers::PropagatorConvolutionBuffers{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Extract buffers
    temp_buffer = buffers.operator_product_buffer
    result_buffer = buffers.backbone_accumulator
    convolution_buffer = buffers.convolution_operand_buffer
    dlr_buffer_1 = buffers.dlr_buffer_1
    dlr_buffer_2 = buffers.dlr_buffer_2

    # calculate the multiplication and convolution from left to right ignoring the two boundary nodes (the matrix multiplications is done in the original order)
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result_buffer, propagator)

    for i in length(edges):-1:2
        edge = edges[i]
        if i == length(edges)
            for hyb in edge.attached_hybs
                apply_hybridization_inplace!(temp_buffer, result_buffer, hyb, model_data, is_reverse=false)
            end
        end
        node = nodes[i]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)
            _multiplication_32!(temp_buffer, result_buffer, op)
            copyto!(result_buffer, temp_buffer)
        end

        # Apply hybridization with is_reverse=true (swaps K+ ↔ K-)
        apply_hybridization_inplace!(temp_buffer, result_buffer, node.attached_hyb, model_data, is_reverse=true)

        # calculate the convolution with the next edge
        edge_minus1 = edges[i-1]
        copyto!(convolution_buffer, propagator)
        for hyb in edge_minus1.attached_hybs
            apply_hybridization_inplace!(temp_buffer, convolution_buffer, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR (in-place) and perform convolution
        # Note: order is reversed compared to forward direction
        tau2dlr_dense!(dlr_buffer_1, dlr_grid, result_buffer, dlr_grid.tau)
        tau2dlr_dense!(dlr_buffer_2, dlr_grid, convolution_buffer, dlr_grid.tau)
        do_convolution_dlr2tau_dense!(result_buffer, model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, dlr_buffer_1, dlr_buffer_2)
    end

    return result_buffer
end

