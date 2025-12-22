"""
Backbone diagram convolution evaluation for dense representation.

Functions for evaluating diagram contributions through multiplication and convolution operations.
These implement the core diagram evaluation logic for propagator and Green's function expansions.
"""

using QMBToolkitDLR
using QMBToolkitED.Dense: DenseOperator

"""
    get_operator(node::BackBoneNode, model_data::ModelData)

Get the DenseOperator for a given node from model data.

# Arguments
- `node::BackBoneNode`: Node containing operator type and indices
- `model_data::ModelData`: Model data with operators

# Returns
Appropriate DenseOperator based on node type:
- `:cdag` with bar_index=-1: Creation operator from exact diagonalization
- `:c` with bar_index=-1: Annihilation operator from exact diagonalization
- `:cdag` with bar_index≠-1: SOE-transformed creation operator
- `:c` with bar_index≠-1: SOE-transformed annihilation operator
"""
function get_operator(node::BackBoneNode, model_data::ModelData)::DenseOperator
    # Cache accessor results for efficiency
    cre_opts = get_cre_opts(model_data)
    ann_opts = get_ann_opts(model_data)
    cre_opts_SOE = get_cre_opts_SOE(model_data)
    ann_opts_SOE = get_ann_opts_SOE(model_data)

    if node.opt_type == :cdag && node.bar_index == -1
        return cre_opts[node.flavor]
    elseif node.opt_type == :c && node.bar_index == -1
        return ann_opts[node.flavor]
    elseif node.opt_type == :cdag && node.bar_index != -1
        return cre_opts_SOE[node.flavor, node.bar_index]
    elseif node.opt_type == :c && node.bar_index != -1
        return ann_opts_SOE[node.flavor, node.bar_index]
    end
end

"""
    multiplication_convolution(nodes, edges, dlr_grid, model_data, propagator, temp_buffer, result_buffer, convolution_buffer)

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
- `temp_buffer::DensePropagator{T}`: Pre-allocated buffer for in-place matrix operations
- `result_buffer::DensePropagator{T}`: Pre-allocated working buffer for accumulated result
- `convolution_buffer::DensePropagator{T}`: Pre-allocated working buffer for convolution operands

# Returns
- `DensePropagator{T}`: Diagram contribution on DLR τ-grid

# Notes
Pre-allocated buffers are reused across all diagram evaluations for efficiency.
"""
function multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T}, propagator::DensePropagator{T}, temp_buffer::DensePropagator{T}, result_buffer::DensePropagator{T}, convolution_buffer::DensePropagator{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")
    # calculate the multiplication and convolution from right to left ignoring the two boundary nodes
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result_buffer, propagator)
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
            op = get_operator(node, model_data)
            _multiplication_23!(temp_buffer, op, result)
            copyto!(result, temp_buffer)
        end

        # Apply hybridization attached to the node
        apply_hybridization_inplace!(temp_buffer, result, node.attached_hyb, model_data, is_reverse=false)

        # calculate the convolution with the next edge
        edge_plus1 = edges[i+1]
        copyto!(convolution_buffer, propagator)
        to_be_convoluted = convolution_buffer
        for hyb in edge_plus1.attached_hybs
            apply_hybridization_inplace!(temp_buffer, to_be_convoluted, hyb, model_data, is_reverse=false)
        end

        result = do_convolution_dlr2tau_dense!(model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, tau2dlr_dense(dlr_grid, to_be_convoluted, dlr_grid.tau), tau2dlr_dense(dlr_grid, result, dlr_grid.tau))
    end
    return result
end


"""
    multiplication_convolution_reverse(nodes, edges, dlr_grid, model_data, propagator, temp_buffer, result_buffer, convolution_buffer)

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
- `DensePropagator{T}`: Diagram contribution on DLR τ-grid
"""
function multiplication_convolution_reverse(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T}, propagator::DensePropagator{T}, temp_buffer::DensePropagator{T}, result_buffer::DensePropagator{T}, convolution_buffer::DensePropagator{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")
    # calculate the multiplication and convolution from left to right ignoring the two boundary nodes (the matrix multiplications is done in the original order)
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    copyto!(result_buffer, propagator)
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
            op = get_operator(node, model_data)
            _multiplication_32!(temp_buffer, result, op)
            copyto!(result, temp_buffer)
        end

        # Apply hybridization with is_reverse=true (swaps K+ ↔ K-)
        apply_hybridization_inplace!(temp_buffer, result, node.attached_hyb, model_data, is_reverse=true)

        # calculate the convolution with the next edge
        edge_minus1 = edges[i-1]
        copyto!(convolution_buffer, propagator)
        to_be_convoluted = convolution_buffer
        for hyb in edge_minus1.attached_hybs
            apply_hybridization_inplace!(temp_buffer, to_be_convoluted, hyb, model_data, is_reverse=false)
        end

        result = do_convolution_dlr2tau_dense!(model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, tau2dlr_dense(dlr_grid, result, dlr_grid.tau), tau2dlr_dense(dlr_grid, to_be_convoluted, dlr_grid.tau))
    end
    return result
end
