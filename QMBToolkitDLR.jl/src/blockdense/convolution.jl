"""
BlockDense convolution operations.

Convolution respects quantum number conservation: only blocks with compatible
quantum numbers contribute to the result.
"""

using Lehmann
using QMBToolkitED.BlockDense

"""
    BlockDenseConvolutionWorkspace{T}

Workspace for block-dense convolution operations.

For each block that needs to be computed in the output, we allocate workspace
for the dense convolution of that block.

Fields:
- `basis::BlockDenseBasis`: The basis
- `flux_result::OperatorFlux`: Flux of the result (F*G has flux = combine_qn(F.flux, G.flux))
- `block_workspaces::Dict{Tuple{Int,Int}, ConvolutionWorkspace{T}}`: Workspace for each output block
"""
struct BlockDenseConvolutionWorkspace{T<:Number}
    basis::BlockDenseBasis
    flux_result::OperatorFlux
    block_workspaces::Dict{Tuple{Int,Int}, ConvolutionWorkspace{T}}
end

"""
    create_blockdense_convolution_workspace(basis::BlockDenseBasis, flux_F::OperatorFlux, flux_G::OperatorFlux, r::Int, n_tau::Int, ::Type{T}) where T<:Number

Create workspace for block-dense convolution.

# Arguments
- `basis`: The block-dense basis
- `flux_F`: Flux of first operator
- `flux_G`: Flux of second operator
- `r`: DLR rank
- `n_tau`: Number of tau points
- `T`: Element type

# Returns
- `BlockDenseConvolutionWorkspace{T}`: Workspace containing ConvolutionWorkspace for each output block
"""
function create_blockdense_convolution_workspace(basis::BlockDenseBasis, flux_F::OperatorFlux, flux_G::OperatorFlux, r::Int, n_tau::Int, ::Type{T}) where T<:Number
    flux_result = OperatorFlux(combine_qn(flux_F.delta_qn, flux_G.delta_qn))
    block_workspaces = Dict{Tuple{Int,Int}, ConvolutionWorkspace{T}}()

    # Determine which output blocks exist
    for (target_idx, target_sector) in enumerate(basis.sectors)
        for (source_idx, source_sector) in enumerate(basis.sectors)
            # Check if this block can exist: target_qn = source_qn + flux_result
            if target_sector.qn == combine_qn(source_sector.qn, flux_result.delta_qn)
                # This block exists in the output
                target_dim = target_sector.dim
                source_dim = source_sector.dim

                # Create workspace for this block
                block_workspaces[(target_idx, source_idx)] = ConvolutionWorkspace{T}(r, target_dim, source_dim, n_tau)
            end
        end
    end

    return BlockDenseConvolutionWorkspace{T}(basis, flux_result, block_workspaces)
end

"""
    do_convolution_dlr2tau_blockdense!(workspace::BlockDenseConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDenseArray{T}, G_dlr::BlockDenseArray{T}) where T<:Number

Compute convolution H = F * G for block-dense arrays in DLR representation.

The convolution respects quantum number conservation. For output block H[α→β],
we sum over all intermediate sectors γ such that the quantum numbers match:
    H[α→β] = Σ_γ F[α→γ] * G[γ→β]

# Arguments
- `workspace`: Pre-allocated workspace for convolution
- `dlr_grid`: DLR grid
- `kernel_cache`: Pre-computed kernel values
- `F_dlr`: First operator in DLR representation
- `G_dlr`: Second operator in DLR representation

# Returns
- `BlockDenseArray{T}`: Result H on tau grid
"""
function do_convolution_dlr2tau_blockdense!(workspace::BlockDenseConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDenseArray{T}, G_dlr::BlockDenseArray{T}) where T<:Number
    H_tau_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    # Compute each output block
    for ((target_idx, source_idx), block_workspace) in workspace.block_workspaces
        # For this output block H[target→source], sum over intermediate sectors
        target_sector = workspace.basis.sectors[target_idx]
        source_sector = workspace.basis.sectors[source_idx]

        H_block_tau = nothing

        for (intermediate_idx, intermediate_sector) in enumerate(workspace.basis.sectors)
            # Check if F[target→intermediate] and G[intermediate→source] blocks exist
            F_key = (target_idx, intermediate_idx)
            G_key = (intermediate_idx, source_idx)

            if haskey(F_dlr.blocks, F_key) && haskey(G_dlr.blocks, G_key)
                # Compute convolution of these blocks
                F_block = F_dlr.blocks[F_key]
                G_block = G_dlr.blocks[G_key]

                # Use dense convolution for this block pair
                block_contrib = do_convolution_dlr2tau!(block_workspace, dlr_grid, kernel_cache, F_block, G_block)

                # Accumulate
                if H_block_tau === nothing
                    H_block_tau = block_contrib
                else
                    H_block_tau .+= block_contrib
                end
            end
        end

        if H_block_tau !== nothing
            H_tau_blocks[(target_idx, source_idx)] = H_block_tau
        end
    end

    return BlockDenseArray{T}(workspace.basis, workspace.flux_result, H_tau_blocks)
end
