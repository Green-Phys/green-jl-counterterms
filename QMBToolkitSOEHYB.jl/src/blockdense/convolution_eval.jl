"""
Convolution evaluation for BlockDense backbone diagrams.

Block-aware implementation that only computes blocks allowed by quantum number conservation.
"""

using QMBToolkitED.BlockDense
using QMBToolkitDLR

"""
    get_operator_matrix_blockdense(node::BackBoneNode, model_data::ModelData{T, BlockDenseFormat}) where T

Get the BlockDense operator matrix for a given node from model data.

# Arguments
- `node::BackBoneNode`: Node containing operator type and indices
- `model_data::ModelData{T, BlockDenseFormat}`: Model data with BlockDense operator matrices

# Returns
Appropriate BlockDenseOperator based on node type:
- `:cdag` with bar_index=-1: Creation operator from exact diagonalization
- `:c` with bar_index=-1: Annihilation operator from exact diagonalization
- `:cdag` with bar_index≠-1: SOE-transformed creation operator
- `:c` with bar_index≠-1: SOE-transformed annihilation operator
"""
function get_operator_matrix_blockdense(node::BackBoneNode, model_data::ModelData{T, BlockDenseFormat}) where T
    if node.opt_type == :cdag && node.bar_index == -1
        return model_data.cre_opts_blockdense[node.flavor]
    elseif node.opt_type == :c && node.bar_index == -1
        return model_data.ann_opts_blockdense[node.flavor]
    elseif node.opt_type == :cdag && node.bar_index != -1
        return model_data.cre_opts_SOE_blockdense[node.flavor, node.bar_index]
    elseif node.opt_type == :c && node.bar_index != -1
        return model_data.ann_opts_SOE_blockdense[node.flavor, node.bar_index]
    end
end

"""
    multiplication_convolution_blockdense(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from right to left.

BlockDense version: respects quantum number conservation and only computes non-zero blocks.

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
- `buffers::PropagatorConvolutionBuffersBlockDense{T}`: Pre-allocated buffers

# Returns
- `BlockDensePropagator{T}`: Backbone contribution on tau grid
"""
function multiplication_convolution_blockdense(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::PropagatorConvolutionBuffersBlockDense{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get basis and flux
    basis = propagator.basis
    flux = propagator.flux

    # Initialize result as copy of propagator
    result_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, block) in propagator.blocks
        result_blocks[key] = copy(block)
    end
    result = BlockDensePropagator{T}(basis, flux, result_blocks)

    # Temporary buffer with same structure as result
    temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, block) in result.blocks
        temp_blocks[key] = similar(block)
    end
    temp_buffer = BlockDensePropagator{T}(basis, result.flux, temp_blocks)

    # Process edges from right to left
    for i in 1:length(edges)-1
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                apply_hybridization_inplace_blockdense!(temp_buffer, result, hyb, model_data, is_reverse=false)
            end
        end

        # Multiply with the next node operator
        node = nodes[i+1]
        if node.opt_type !== :empty
            op_matrix = get_operator_matrix_blockdense(node, model_data)

            # Need to create temp buffer with correct flux for operator multiplication
            op_flux = OperatorFlux(combine_qn(op_matrix.flux.delta_qn, result.flux.delta_qn))

            # Determine which blocks will exist in the product: op_matrix * result
            # Result block [tgt, src] exists if there exists mid such that:
            #   op_matrix has block [tgt, mid] AND result has block [mid, src]
            op_result_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
            n_tau = length(model_data.dlr_grid.tau)
            for ((tgt, mid), op_block) in op_matrix.blocks
                for ((mid2, src), result_block) in result.blocks
                    if mid == mid2
                        key = (tgt, src)
                        if !haskey(op_result_blocks, key)
                            dim_tgt = basis.sectors[tgt].dim
                            dim_src = basis.sectors[src].dim
                            op_result_blocks[key] = zeros(T, n_tau, dim_tgt, dim_src)
                        end
                    end
                end
            end
            op_temp_buffer = BlockDensePropagator{T}(basis, op_flux, op_result_blocks)

            _multiplication_23_blockdense!(op_temp_buffer, op_matrix, result)

            # Update result structure to match op_temp_buffer
            result = BlockDensePropagator{T}(basis, op_flux, Dict(k => copy(v) for (k,v) in op_temp_buffer.blocks))

            # Update temp_buffer to match new flux
            temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
            for (key, block) in result.blocks
                temp_blocks[key] = similar(block)
            end
            temp_buffer = BlockDensePropagator{T}(basis, result.flux, temp_blocks)
        end

        # Apply hybridization attached to the node
        apply_hybridization_inplace_blockdense!(temp_buffer, result, node.attached_hyb, model_data, is_reverse=false)

        # Calculate the convolution with the next edge
        edge_plus1 = edges[i+1]

        # Create convolution_buffer as copy of propagator
        convolution_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
        for (key, block) in propagator.blocks
            convolution_blocks[key] = copy(block)
        end
        to_be_convoluted = BlockDensePropagator{T}(basis, flux, convolution_blocks)

        # Create temp buffer for convolution hybridizations
        conv_temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
        for (key, block) in to_be_convoluted.blocks
            conv_temp_blocks[key] = similar(block)
        end
        conv_temp_buffer = BlockDensePropagator{T}(basis, flux, conv_temp_blocks)

        for hyb in edge_plus1.attached_hybs
            apply_hybridization_inplace_blockdense!(conv_temp_buffer, to_be_convoluted, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR and perform convolution
        to_be_convoluted_dlr = tau2dlr_blockdense(dlr_grid, to_be_convoluted, dlr_grid.tau)
        result_dlr = tau2dlr_blockdense(dlr_grid, result, dlr_grid.tau)

        # Create convolution workspace
        r = dlr_grid.size
        n_tau = length(dlr_grid.tau)
        conv_workspace = create_blockdense_convolution_workspace(basis, to_be_convoluted_dlr.flux, result_dlr.flux, r, n_tau, T)

        result = do_convolution_dlr2tau_blockdense!(conv_workspace, dlr_grid, model_data.kernel_cache, to_be_convoluted_dlr, result_dlr)

        # Update temp_buffer to match new result structure
        temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
        for (key, block) in result.blocks
            temp_blocks[key] = similar(block)
        end
        temp_buffer = BlockDensePropagator{T}(basis, result.flux, temp_blocks)
    end

    return result
end

"""
    multiplication_convolution_reverse_blockdense(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from left to right.

BlockDense version with swapped K± hybridizations.

# Key Differences from Forward Direction
- Traverses from left to right (indices M down to 1)
- Uses `is_reverse=true` for node-attached hybridizations (swaps K+ ↔ K-)
- Uses `_multiplication_32_blockdense!` instead of `_multiplication_23_blockdense!`
- Convolution operand order is reversed: result first, then to_be_convoluted
"""
function multiplication_convolution_reverse_blockdense(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::PropagatorConvolutionBuffersBlockDense{T}) where {T<:Number}
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get basis and flux
    basis = propagator.basis
    flux = propagator.flux

    # Initialize result as copy of propagator
    result_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, block) in propagator.blocks
        result_blocks[key] = copy(block)
    end
    result = BlockDensePropagator{T}(basis, flux, result_blocks)

    # Temporary buffer with same structure as result
    temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, block) in result.blocks
        temp_blocks[key] = similar(block)
    end
    temp_buffer = BlockDensePropagator{T}(basis, result.flux, temp_blocks)

    # Process edges from left to right
    for i in length(edges):-1:2
        edge = edges[i]
        if i == length(edges)
            for hyb in edge.attached_hybs
                apply_hybridization_inplace_blockdense!(temp_buffer, result, hyb, model_data, is_reverse=false)
            end
        end

        node = nodes[i]
        if node.opt_type !== :empty
            op_matrix = get_operator_matrix_blockdense(node, model_data)

            # Need to create temp buffer with correct flux for operator multiplication
            op_flux = OperatorFlux(combine_qn(result.flux.delta_qn, op_matrix.flux.delta_qn))

            # Determine which blocks will exist in the product: result * op_matrix
            # Result block [tgt, src] exists if there exists mid such that:
            #   result has block [tgt, mid] AND op_matrix has block [mid, src]
            op_result_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
            n_tau = length(model_data.dlr_grid.tau)
            for ((tgt, mid), result_block) in result.blocks
                for ((mid2, src), op_block) in op_matrix.blocks
                    if mid == mid2
                        key = (tgt, src)
                        if !haskey(op_result_blocks, key)
                            dim_tgt = basis.sectors[tgt].dim
                            dim_src = basis.sectors[src].dim
                            op_result_blocks[key] = zeros(T, n_tau, dim_tgt, dim_src)
                        end
                    end
                end
            end
            op_temp_buffer = BlockDensePropagator{T}(basis, op_flux, op_result_blocks)

            _multiplication_32_blockdense!(op_temp_buffer, result, op_matrix)

            # Update result structure
            result = BlockDensePropagator{T}(basis, op_flux, Dict(k => copy(v) for (k,v) in op_temp_buffer.blocks))

            # Update temp_buffer to match new flux
            temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
            for (key, block) in result.blocks
                temp_blocks[key] = similar(block)
            end
            temp_buffer = BlockDensePropagator{T}(basis, result.flux, temp_blocks)
        end

        # Apply hybridization with is_reverse=true (swaps K+ ↔ K-)
        apply_hybridization_inplace_blockdense!(temp_buffer, result, node.attached_hyb, model_data, is_reverse=true)

        # Calculate the convolution with the next edge
        edge_minus1 = edges[i-1]

        # Create convolution_buffer as copy of propagator
        convolution_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
        for (key, block) in propagator.blocks
            convolution_blocks[key] = copy(block)
        end
        to_be_convoluted = BlockDensePropagator{T}(basis, flux, convolution_blocks)

        # Create temp buffer for convolution hybridizations
        conv_temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
        for (key, block) in to_be_convoluted.blocks
            conv_temp_blocks[key] = similar(block)
        end
        conv_temp_buffer = BlockDensePropagator{T}(basis, flux, conv_temp_blocks)

        for hyb in edge_minus1.attached_hybs
            apply_hybridization_inplace_blockdense!(conv_temp_buffer, to_be_convoluted, hyb, model_data, is_reverse=false)
        end

        # Transform to DLR and perform convolution (reversed order)
        result_dlr = tau2dlr_blockdense(dlr_grid, result, dlr_grid.tau)
        to_be_convoluted_dlr = tau2dlr_blockdense(dlr_grid, to_be_convoluted, dlr_grid.tau)

        # Create convolution workspace
        r = dlr_grid.size
        n_tau = length(dlr_grid.tau)
        conv_workspace = create_blockdense_convolution_workspace(basis, result_dlr.flux, to_be_convoluted_dlr.flux, r, n_tau, T)

        result = do_convolution_dlr2tau_blockdense!(conv_workspace, dlr_grid, model_data.kernel_cache, result_dlr, to_be_convoluted_dlr)

        # Update temp_buffer to match new result structure
        temp_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
        for (key, block) in result.blocks
            temp_blocks[key] = similar(block)
        end
        temp_buffer = BlockDensePropagator{T}(basis, result.flux, temp_blocks)
    end

    return result
end
