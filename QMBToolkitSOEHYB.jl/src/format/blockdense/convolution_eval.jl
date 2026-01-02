"""
Backbone diagram evaluation for BlockDense representation.

Block-aware implementation that only computes blocks allowed by quantum number conservation.
Uses pre-allocated flux-indexed buffer pools from BlockDenseBackboneBuffers.

Note: get_operator is defined in diagram/backbone_common.jl as a format-agnostic function.
"""

using QMBToolkitED.BlockDense
using QMBToolkitDLR

"""
    multiplication_convolution_blockdense(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from right to left.
Uses pre-allocated buffers from BlockDenseBackboneBuffers.

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
- `dlr_grid::DLRGrid`: DLR grid
- `model_data::ModelData{T, BlockDenseFormat}`: Model data
- `propagator::BlockDensePropagator{T}`: Propagator (bare or bold)
- `buffers::BlockDenseBackboneBuffers{T}`: Pre-allocated work buffers

# Returns
- `BlockDensePropagator{T}`: Backbone contribution on tau grid (reference to buffer)

# Buffer Usage (7 roles)
- `result`: Accumulates backbone result (from result_pool, indexed by flux)
- `hyb_temp`: Scratch for hybridization on result (from hyb_temp_pool, indexed by flux)
- `op_product`: Holds operator multiplication output (from op_product_pool, indexed by new flux)
- `conv_operand`: Propagator copy for convolution (fixed zero flux)
- `conv_hyb_temp`: Scratch for hybridization on conv operand (fixed zero flux)
- `conv_dlr`: DLR transform of conv operand (fixed zero flux)
- `result_dlr`: DLR transform of result (from result_dlr_pool, indexed by flux)
"""
function multiplication_convolution_blockdense(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::BlockDenseBackboneBuffers{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get initial flux
    current_flux = propagator.flux

    # Get result buffer for current flux and copy propagator into it
    result = get_result(buffers, current_flux)
    copyto!(result, propagator)

    # Get hyb_temp buffer for current flux
    hyb_temp = get_hyb_temp(buffers, current_flux)

    # Process edges from right to left
    for i in 1:length(edges)-1
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                apply_hybridization_inplace_blockdense!(hyb_temp, result, hyb, model_data, is_reverse=false)
            end
        end

        # Multiply with the next node operator
        node = nodes[i+1]
        if node.opt_type !== :empty
            op_matrix = get_operator(node, model_data)

            # Compute new flux after operator multiplication
            new_flux = OperatorFlux(combine_qn(op_matrix.flux.delta_qn, current_flux.delta_qn))

            # Get op_product buffer for the new flux
            op_product = get_op_product(buffers, new_flux)
            fill!(op_product, zero(T))

            # Perform multiplication: op_matrix * result -> op_product
            _multiplication_23_blockdense!(op_product, op_matrix, result)

            # Update current flux and get new result buffer
            current_flux = new_flux
            result = get_result(buffers, current_flux)
            copyto!(result, op_product)

            # Update hyb_temp to match new flux
            hyb_temp = get_hyb_temp(buffers, current_flux)
        end

        # Apply hybridization attached to the node
        apply_hybridization_inplace_blockdense!(hyb_temp, result, node.attached_hyb, model_data, is_reverse=false)

        # Calculate the convolution with the next edge
        edge_plus1 = edges[i+1]

        # Get conv_operand buffer (fixed zero flux) and copy propagator
        conv_operand = get_conv_operand(buffers)
        copyto!(conv_operand, propagator)

        # Get conv_hyb_temp buffer (fixed zero flux)
        conv_hyb_temp = get_conv_hyb_temp(buffers)

        for hyb in edge_plus1.attached_hybs
            apply_hybridization_inplace_blockdense!(conv_hyb_temp, conv_operand, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR using buffers
        conv_dlr = get_conv_dlr(buffers)
        tau2dlr_blockdense!(conv_dlr, dlr_grid, conv_operand, dlr_grid.tau)

        result_dlr = get_result_dlr(buffers, current_flux)
        tau2dlr_blockdense!(result_dlr, dlr_grid, result, dlr_grid.tau)

        # Perform convolution - result is written back to result buffer
        do_convolution_dlr2tau_blockdense!(result, model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, conv_dlr, result_dlr)
    end

    return result
end

"""
    multiplication_convolution_reverse_blockdense(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from left to right.
Uses pre-allocated buffers from BlockDenseBackboneBuffers.

# Key Differences from Forward Direction
- Traverses from left to right (indices M down to 1)
- Uses `is_reverse=true` for node-attached hybridizations (swaps K+ ↔ K-)
- Uses `_multiplication_32_blockdense!` instead of `_multiplication_23_blockdense!`
- Convolution operand order is reversed: result first, then to_be_convoluted
"""
function multiplication_convolution_reverse_blockdense(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::BlockDenseBackboneBuffers{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get initial flux
    current_flux = propagator.flux

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
                apply_hybridization_inplace_blockdense!(hyb_temp, result, hyb, model_data, is_reverse=false)
            end
        end

        node = nodes[i]
        if node.opt_type !== :empty
            op_matrix = get_operator(node, model_data)

            # Compute new flux after operator multiplication
            new_flux = OperatorFlux(combine_qn(current_flux.delta_qn, op_matrix.flux.delta_qn))

            # Get op_product buffer for the new flux
            op_product = get_op_product(buffers, new_flux)
            fill!(op_product, zero(T))

            # Perform multiplication: result * op_matrix -> op_product
            _multiplication_32_blockdense!(op_product, result, op_matrix)

            # Update current flux and get new result buffer
            current_flux = new_flux
            result = get_result(buffers, current_flux)
            copyto!(result, op_product)

            # Update hyb_temp to match new flux
            hyb_temp = get_hyb_temp(buffers, current_flux)
        end

        # Apply hybridization with is_reverse=true (swaps K+ ↔ K-)
        apply_hybridization_inplace_blockdense!(hyb_temp, result, node.attached_hyb, model_data, is_reverse=true)

        # Calculate the convolution with the next edge
        edge_minus1 = edges[i-1]

        # Get conv_operand buffer (fixed zero flux) and copy propagator
        conv_operand = get_conv_operand(buffers)
        copyto!(conv_operand, propagator)

        # Get conv_hyb_temp buffer (fixed zero flux)
        conv_hyb_temp = get_conv_hyb_temp(buffers)

        for hyb in edge_minus1.attached_hybs
            apply_hybridization_inplace_blockdense!(conv_hyb_temp, conv_operand, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR using buffers (note: order reversed)
        result_dlr = get_result_dlr(buffers, current_flux)
        tau2dlr_blockdense!(result_dlr, dlr_grid, result, dlr_grid.tau)

        conv_dlr = get_conv_dlr(buffers)
        tau2dlr_blockdense!(conv_dlr, dlr_grid, conv_operand, dlr_grid.tau)

        # Perform convolution (reversed order) - result is written back to result buffer
        do_convolution_dlr2tau_blockdense!(result, model_data.convolution_workspace, dlr_grid, model_data.kernel_cache, result_dlr, conv_dlr)
    end

    return result
end

#=============================================================================
# Interface Implementations
=============================================================================#

# multiplication_convolution - interface aliases
multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::BlockDenseBackboneBuffers{T}) where T = multiplication_convolution_blockdense(nodes, edges, dlr_grid, model_data, propagator, buffers)

multiplication_convolution_reverse(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::BlockDenseBackboneBuffers{T}) where T = multiplication_convolution_reverse_blockdense(nodes, edges, dlr_grid, model_data, propagator, buffers)
