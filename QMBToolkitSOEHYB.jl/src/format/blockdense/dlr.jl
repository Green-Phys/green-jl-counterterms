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
function tau2dlr_blockdense(dlr_grid::DLRGrid, prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where {T}
    blocks_dlr = Dict{Tuple{Int, Int}, Array{T, 3}}()

    for ((target, source), block_tau) in prop.blocks
        # Transform this block: (n_tau, n_row, n_col) -> (r, n_row, n_col)
        block_dlr = QMBToolkitDLR.tau2dlr(dlr_grid, block_tau, tau_grid)
        blocks_dlr[(target, source)] = block_dlr
    end

    return BlockDensePropagator{T}(prop.basis, prop.flux, blocks_dlr)
end

"""
    tau2dlr_blockdense!(dest::BlockDensePropagator{T}, dlr_grid::DLRGrid, prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T

In-place transform block-dense propagator from tau space to DLR representation.
Writes result to dest, which must have the same block structure as prop.
"""
function tau2dlr_blockdense!(dest::BlockDensePropagator{T}, dlr_grid::DLRGrid,
        prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where {T}
    @boundscheck @assert dest.basis === prop.basis "Destination and source must have same basis"
    @boundscheck @assert dest.flux.delta_qn == prop.flux.delta_qn "Destination and source must have same flux"

    for ((target, source), block_tau) in prop.blocks
        @boundscheck @assert haskey(dest.blocks, (target, source)) "Destination missing block ($target, $source)"
        # Transform this block: (n_tau, n_row, n_col) -> (r, n_row, n_col)
        block_dlr = QMBToolkitDLR.tau2dlr(dlr_grid, block_tau, tau_grid)
        copyto!(dest.blocks[(target, source)], block_dlr)
    end

    return dest
end

"""
    dlr2tau_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T

Transform block-dense propagator from DLR representation to tau space.
Each block is transformed independently using standard dlr2tau.
"""
function dlr2tau_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where {T}
    blocks_tau = Dict{Tuple{Int, Int}, Array{T, 3}}()

    for ((target, source), block_dlr) in prop_dlr.blocks
        # Transform this block: (r, n_row, n_col) -> (n_tau, n_row, n_col)
        block_tau = QMBToolkitDLR.dlr2tau(dlr_grid, block_dlr, tau_grid)
        blocks_tau[(target, source)] = block_tau
    end

    return BlockDensePropagator{T}(prop_dlr.basis, prop_dlr.flux, blocks_tau)
end

#=============================================================================
# Reflection Operations
=============================================================================#

"""
    do_reflection_tau2tau_blockdense!(dest::BlockDensePropagator{T}, dlr_grid::DLRGrid, prop_tau::BlockDensePropagator{T}) where T

In-place reflection operation on tau-space BlockDense data: dest = prop_tau(beta - tau).
Writes result to dest's existing blocks. Assumes dest has matching block structure and flux.
"""
function do_reflection_tau2tau_blockdense!(dest::BlockDensePropagator{T}, dlr_grid::DLRGrid,
        prop_tau::BlockDensePropagator{T}) where {T}
    # Transform to DLR
    prop_dlr = tau2dlr_blockdense(dlr_grid, prop_tau, dlr_grid.τ)

    # Evaluate at reflected tau points (beta - tau)
    tau_grid_reflected = dlr_grid.β .- dlr_grid.τ

    for ((target, source), block_dlr) in prop_dlr.blocks
        block_tau = QMBToolkitDLR.dlr2tau(dlr_grid, block_dlr, tau_grid_reflected)
        copyto!(dest.blocks[(target, source)], block_tau)
    end

    return dest
end

#=============================================================================
# Convolution Operations
=============================================================================#

"""
    do_convolution_dlr2tau_blockdense!(dest::BlockDensePropagator{T}, workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDensePropagator{T}, G_dlr::BlockDensePropagator{T}) where T

Compute convolution H = F * G for block-dense propagators, writing result to dest.
In-place version that reuses pre-allocated destination buffer.

The destination must have the correct flux (flux_F + flux_G) and matching block structure.
"""
function do_convolution_dlr2tau_blockdense!(
        dest::BlockDensePropagator{T}, workspace::ConvolutionWorkspace{T},
        dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDensePropagator{T},
        G_dlr::BlockDensePropagator{T}) where {T <: Number}
    # Get basis from F_dlr (both should have the same basis)
    basis = F_dlr.basis

    # Compute expected result flux
    flux_result = OperatorFlux(combine_qn(F_dlr.flux.delta_qn, G_dlr.flux.delta_qn))
    @boundscheck @assert dest.flux.delta_qn == flux_result.delta_qn "Destination has wrong flux"

    # Zero out destination blocks
    fill!(dest, zero(T))

    # Compute each output block
    @inbounds for (target_idx, target_sector) in enumerate(basis.sectors)
        for (source_idx, source_sector) in enumerate(basis.sectors)
            # Check if this output block can exist given the result flux
            if target_sector.qn != combine_qn(source_sector.qn, flux_result.delta_qn)
                continue
            end

            dest_key = (target_idx, source_idx)
            if !haskey(dest.blocks, dest_key)
                continue  # Skip if destination doesn't have this block
            end

            # Sum over all intermediate sectors
            for (intermediate_idx, intermediate_sector) in enumerate(basis.sectors)
                F_key = (target_idx, intermediate_idx)
                G_key = (intermediate_idx, source_idx)

                if haskey(F_dlr.blocks, F_key) && haskey(G_dlr.blocks, G_key)
                    F_block = F_dlr.blocks[F_key]
                    G_block = G_dlr.blocks[G_key]

                    # Use dense convolution with do_view=true since workspace is max-sized
                    block_contrib = QMBToolkitDLR.do_convolution_dlr2tau!(
                        workspace, dlr_grid, kernel_cache, F_block, G_block, do_view = true)

                    dest.blocks[dest_key] .+= block_contrib
                end
            end
        end
    end

    return dest
end

#=============================================================================
# Tau Extraction Operations
=============================================================================#

"""
    extract_at_tau_blockdense(propagator::BlockDensePropagator{T}, tau::Float64, dlr_grid::DLRGrid) where T

Extract propagator value at specific tau by DLR interpolation, return as dense matrix.
"""
function extract_at_tau_blockdense(propagator::BlockDensePropagator{T}, tau::Float64, dlr_grid::DLRGrid) where {T}
    # Convert to DLR coefficients
    prop_dlr = tau2dlr_blockdense(dlr_grid, propagator, dlr_grid.τ)

    # Evaluate at the requested tau
    prop_at_tau = dlr2tau_blockdense(dlr_grid, prop_dlr, [tau])

    # Convert single-time-slice BlockDensePropagator to dense matrix
    return blockdense_to_dense_single_tau(prop_at_tau)
end

"""
    blockdense_to_dense_single_tau(bda::BlockDensePropagator{T}) where T

Convert a BlockDensePropagator with single time slice (n_tau=1) to a dense matrix.
"""
function blockdense_to_dense_single_tau(bda::BlockDensePropagator{T}) where {T}
    # Determine total dimension from basis
    n_total = sum(sector.dim for sector in bda.basis.sectors)
    result = zeros(T, n_total, n_total)

    # Fill in blocks
    for (key, block) in bda.blocks
        (i, j) = key
        dim_i = bda.basis.sectors[i].dim
        dim_j = bda.basis.sectors[j].dim

        # Get row and column offsets
        row_offset = sum(bda.basis.sectors[k].dim for k in 1:(i - 1); init = 0)
        col_offset = sum(bda.basis.sectors[k].dim for k in 1:(j - 1); init = 0)

        # Extract single time slice (should be at index 1)
        @boundscheck @assert size(block, 1) == 1 "Expected single time slice"
        result[(row_offset + 1):(row_offset + dim_i), (col_offset + 1):(col_offset + dim_j)] = block[1, :, :]
    end

    return result
end

#=============================================================================
# Interface Implementations
=============================================================================#

# DLR transformations - interface versions
function tau2dlr(dlr_grid::DLRGrid, prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where {T}
    tau2dlr_blockdense(dlr_grid, prop, tau_grid)
end
function tau2dlr!(dest::BlockDensePropagator{T}, dlr_grid::DLRGrid,
        prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where {T}
    tau2dlr_blockdense!(dest, dlr_grid, prop, tau_grid)
end
function dlr2tau(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where {T}
    dlr2tau_blockdense(dlr_grid, prop_dlr, tau_grid)
end

# Reflection (in-place only)
function do_reflection_tau2tau!(dest::BlockDensePropagator{T}, dlr_grid::DLRGrid,
        prop_tau::BlockDensePropagator{T}) where {T}
    do_reflection_tau2tau_blockdense!(dest, dlr_grid, prop_tau)
end

# Extract at tau
function extract_at_tau(propagator::BlockDensePropagator{T}, tau::Float64, dlr_grid::DLRGrid) where {T}
    extract_at_tau_blockdense(propagator, tau, dlr_grid)
end

# Convolution
function do_convolution_dlr2tau!(dest::BlockDensePropagator{T}, workspace::ConvolutionWorkspace{T},
        dlr_grid::DLRGrid, kernel_cache::KernelCache,
        F_dlr::BlockDensePropagator{T}, G_dlr::BlockDensePropagator{T}) where {T}
    do_convolution_dlr2tau_blockdense!(dest, workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)
end
