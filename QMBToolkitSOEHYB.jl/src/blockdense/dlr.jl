"""
DLR operations for BlockDensePropagator.

Provides tau2dlr, dlr2tau, reflection, and convolution operations for block-dense propagators.
Each block is transformed/operated independently, preserving block structure.
"""

using QMBToolkitDLR
using QMBToolkitED.BlockDense

# BlockDensePropagator is defined in BlockDensePropagator.jl

#=============================================================================
# DLR Transformations
=============================================================================#

"""
    tau2dlr_blockdense(dlr_grid::DLRGrid, prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T

Transform block-dense propagator from tau space to DLR representation.
Each block is transformed independently using standard tau2dlr.
"""
function tau2dlr_blockdense(dlr_grid::DLRGrid, prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T
    blocks_dlr = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_tau) in prop.blocks
        # Transform this block: (n_tau, n_row, n_col) -> (r, n_row, n_col)
        block_dlr = tau2dlr(dlr_grid, block_tau, tau_grid)
        blocks_dlr[(target, source)] = block_dlr
    end

    return BlockDensePropagator{T}(prop.basis, prop.flux, blocks_dlr)
end

"""
    dlr2tau_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T

Transform block-dense propagator from DLR representation to tau space.
Each block is transformed independently using standard dlr2tau.
"""
function dlr2tau_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T
    blocks_tau = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_dlr) in prop_dlr.blocks
        # Transform this block: (r, n_row, n_col) -> (n_tau, n_row, n_col)
        block_tau = dlr2tau(dlr_grid, block_dlr, tau_grid)
        blocks_tau[(target, source)] = block_tau
    end

    return BlockDensePropagator{T}(prop_dlr.basis, prop_dlr.flux, blocks_tau)
end

#=============================================================================
# Reflection Operations
=============================================================================#

"""
    do_reflection_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}) where T

Perform reflection operation on block-dense propagator: F(tau) = H(beta - tau).
Each block is reflected independently using standard DLR reflection.
"""
function do_reflection_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}) where T
    tau_grid_reflected = dlr_grid.beta .- dlr_grid.tau
    blocks_tau = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_dlr) in prop_dlr.blocks
        # Reflect this block
        block_tau = dlr2tau(dlr_grid, block_dlr, tau_grid_reflected)
        blocks_tau[(target, source)] = block_tau
    end

    return BlockDensePropagator{T}(prop_dlr.basis, prop_dlr.flux, blocks_tau)
end

"""
    do_reflection_tau2tau_blockdense(dlr_grid::DLRGrid, prop_tau::BlockDensePropagator{T}) where T

Perform reflection operation on tau-space BlockDense data: F(tau) = H(beta - tau).
First transforms to DLR, then applies reflection.
"""
function do_reflection_tau2tau_blockdense(dlr_grid::DLRGrid, prop_tau::BlockDensePropagator{T}) where T
    prop_dlr = tau2dlr_blockdense(dlr_grid, prop_tau, dlr_grid.tau)
    return do_reflection_blockdense(dlr_grid, prop_dlr)
end

#=============================================================================
# Convolution Operations
=============================================================================#

"""
    do_convolution_dlr2tau_blockdense!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDensePropagator{T}, G_dlr::BlockDensePropagator{T}) where T

Compute convolution H = F * G for block-dense propagators using pre-allocated workspace.

Uses a single ConvolutionWorkspace with `do_view=true` to handle all block sizes.
The result flux is computed as flux_F + flux_G. Only blocks that have non-zero contributions
(i.e., where matching F and G blocks exist) are included in the output.
"""
function do_convolution_dlr2tau_blockdense!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDensePropagator{T}, G_dlr::BlockDensePropagator{T}) where T<:Number
    H_tau_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    # Get basis from F_dlr (both should have the same basis)
    basis = F_dlr.basis

    # Compute result flux
    flux_result = OperatorFlux(combine_qn(F_dlr.flux.delta_qn, G_dlr.flux.delta_qn))

    # Compute each output block
    for (target_idx, target_sector) in enumerate(basis.sectors)
        for (source_idx, source_sector) in enumerate(basis.sectors)
            # Check if this output block can exist given the result flux
            if target_sector.qn != combine_qn(source_sector.qn, flux_result.delta_qn)
                continue
            end

            H_block_tau = nothing

            # Sum over all intermediate sectors
            for (intermediate_idx, intermediate_sector) in enumerate(basis.sectors)
                F_key = (target_idx, intermediate_idx)
                G_key = (intermediate_idx, source_idx)

                if haskey(F_dlr.blocks, F_key) && haskey(G_dlr.blocks, G_key)
                    F_block = F_dlr.blocks[F_key]
                    G_block = G_dlr.blocks[G_key]

                    # Use dense convolution with do_view=true since workspace is max-sized
                    block_contrib = do_convolution_dlr2tau!(workspace, dlr_grid, kernel_cache, F_block, G_block, do_view=true)

                    if H_block_tau === nothing
                        H_block_tau = copy(block_contrib)
                    else
                        H_block_tau .+= block_contrib
                    end
                end
            end

            if H_block_tau !== nothing
                H_tau_blocks[(target_idx, source_idx)] = H_block_tau
            end
        end
    end

    return BlockDensePropagator{T}(basis, flux_result, H_tau_blocks)
end
