"""
DLR operations for Propagator.

Provides tau2dlr, dlr2tau, reflection, and convolution operations for propagators.
Each block is transformed/operated independently, preserving block structure.
Layout: (dim1, dim2, n_tau) for tau-space, (dim1, dim2, r) for DLR representation.
"""

using QMBToolkitDLR
using Lehmann
using QMBToolkitED: Basis

#=============================================================================
# DLR Transformations
=============================================================================#

"""
    tau2dlr_propagator(dlr_grid::DLRGrid, prop::Propagator{T}, tau_grid::Vector{Float64}) where T

Transform propagator from tau space to DLR representation.
Each block is transformed independently using standard tau2dlr.
"""
function tau2dlr_propagator(dlr_grid::DLRGrid, prop::Propagator{T}, tau_grid::Vector{Float64}) where {T}
    blocks_dlr = Dict{Tuple{Int, Int}, Array{T, 3}}()

    for ((target, source), block_tau) in prop.blocks
        block_dlr = QMBToolkitDLR.tau2dlr(dlr_grid, block_tau, tau_grid; axis = 3)
        blocks_dlr[(target, source)] = block_dlr
    end

    return Propagator{T}(prop.basis, prop.flux, blocks_dlr)
end

"""
    tau2dlr_propagator!(dest::Propagator{T}, dlr_grid::DLRGrid, prop::Propagator{T}, tau_grid::Vector{Float64}, dlr_workspace::DLRWorkspace{T}) where T

In-place transform propagator from tau space to DLR representation.
Writes result to dest, which must have the same block structure as prop.
Uses Lehmann.tau2dlr! for efficient transformation when workspace is large enough.
"""
function tau2dlr_propagator!(dest::Propagator{T}, dlr_grid::DLRGrid,
        prop::Propagator{T}, tau_grid::Vector{Float64}, dlr_workspace::DLRWorkspace{T}) where {T}
    @boundscheck @assert dest.basis === prop.basis "Destination and source must have same basis"
    @boundscheck @assert dest.flux == prop.flux "Destination and source must have same flux"

    n_dlr = length(dlr_grid)

    for ((target, source), block_tau) in prop.blocks
        @boundscheck @assert haskey(dest.blocks, (target, source)) "Destination missing block ($target, $source)"
        dim1, dim2, _ = size(block_tau)
        dest_block = dest.blocks[(target, source)]

        # Use in-place Lehmann.tau2dlr! if workspace is large enough
        if dlr_workspace.n_dlr >= n_dlr && dlr_workspace.n_other >= dim1 * dim2
            Lehmann.tau2dlr!(dest_block, dlr_grid, block_tau, dlr_workspace)
        else
            # Fallback to allocating version for blocks larger than workspace
            block_dlr = QMBToolkitDLR.tau2dlr(dlr_grid, block_tau, tau_grid; axis = 3)
            copyto!(dest_block, block_dlr)
        end
    end

    return dest
end

"""
    dlr2tau_propagator(dlr_grid::DLRGrid, prop_dlr::Propagator{T}, tau_grid::Vector{Float64}) where T

Transform propagator from DLR representation to tau space.
Each block is transformed independently using standard dlr2tau.
"""
function dlr2tau_propagator(dlr_grid::DLRGrid, prop_dlr::Propagator{T}, tau_grid::Vector{Float64}) where {T}
    blocks_tau = Dict{Tuple{Int, Int}, Array{T, 3}}()

    for ((target, source), block_dlr) in prop_dlr.blocks
        block_tau = QMBToolkitDLR.dlr2tau(dlr_grid, block_dlr, tau_grid; axis = 3)
        blocks_tau[(target, source)] = block_tau
    end

    return Propagator{T}(prop_dlr.basis, prop_dlr.flux, blocks_tau)
end

#=============================================================================
# Reflection Operations
=============================================================================#

"""
    do_reflection_tau2tau_propagator!(dest::Propagator{T}, prop_tau::Propagator{T},
                                      reflection_matrix::AbstractMatrix{Float64},
                                      dlr_workspace::DLRWorkspace{T}) where T

In-place reflection operation using pre-computed reflection matrix: dest = prop_tau(β - τ).
Uses direct matrix multiply R * src instead of tau2dlr + dlr2tau, saving one LU solve.

# Arguments

  - `dest`: Output propagator (must have same structure as prop_tau)
  - `prop_tau`: Input propagator in tau space
  - `reflection_matrix`: Pre-computed R = K(β-τ,ω) * inv(K(τ,ω)), from ConvolutionWorkspace
  - `dlr_workspace`: Workspace for buffer allocation

# Performance

This version saves one LU solve compared to the tau2dlr + dlr2tau approach,
providing ~20-30% speedup in the reflection step.
"""
function do_reflection_tau2tau_propagator!(dest::Propagator{T}, prop_tau::Propagator{T},
        reflection_matrix::AbstractMatrix{Float64},
        dlr_workspace::DLRWorkspace{T}) where {T}
    @boundscheck @assert dest.basis === prop_tau.basis "Destination and source must have same basis"
    @boundscheck @assert dest.flux == prop_tau.flux "Destination and source must have same flux"

    n_tau = size(reflection_matrix, 1)

    for ((target, source), block_tau) in prop_tau.blocks
        @boundscheck @assert haskey(dest.blocks, (target, source)) "Destination missing block ($target, $source)"

        dest_block = dest.blocks[(target, source)]
        dim1, dim2, _ = size(block_tau)

        # Use direct reflection matrix multiply if workspace is large enough
        if dlr_workspace.n_dlr >= n_tau && dlr_workspace.n_other >= dim1 * dim2
            QMBToolkitDLR.do_reflection_tau2tau!(dest_block, block_tau, reflection_matrix, dlr_workspace)
        else
            # Fallback to allocating version for blocks larger than workspace
            # This should rarely happen in practice
            buffer_in = reshape(permutedims(block_tau, (3, 1, 2)), n_tau, dim1 * dim2)
            buffer_out = reflection_matrix * buffer_in
            permutedims!(dest_block, reshape(buffer_out, n_tau, dim1, dim2), (2, 3, 1))
        end
    end

    return dest
end

#=============================================================================
# Convolution Operations
=============================================================================#

"""
    do_convolution_dlr2tau_propagator!(dest::Propagator{T}, workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Propagator{T}, G_dlr::Propagator{T}) where T

Compute convolution H = F * G for propagators, writing result to dest.
In-place version that reuses pre-allocated destination buffer.

The destination must have the correct flux (flux_F + flux_G) and matching block structure.
"""
function do_convolution_dlr2tau_propagator!(
        dest::Propagator{T}, workspace::ConvolutionWorkspace{T},
        dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Propagator{T},
        G_dlr::Propagator{T}) where {T <: Number}
    # Get basis from F_dlr (both should have the same basis)
    basis = F_dlr.basis

    # Compute expected result flux
    flux_result = F_dlr.flux + G_dlr.flux
    @boundscheck @assert dest.flux == flux_result "Destination has wrong flux"

    # Zero out destination blocks
    fill!(dest, zero(T))

    # Compute each output block
    @inbounds for (target_idx, target_sector) in enumerate(basis.sectors)
        for (source_idx, source_sector) in enumerate(basis.sectors)
            # Check if this output block can exist given the result flux
            if target_sector.qn != source_sector.qn + flux_result
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

                    # Add to destination
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
    extract_at_tau_propagator(propagator::Propagator{T}, tau::Float64, dlr_grid::DLRGrid) where T

Extract propagator value at specific tau by DLR interpolation, return as dense matrix.
"""
function extract_at_tau_propagator(propagator::Propagator{T}, tau::Float64, dlr_grid::DLRGrid) where {T}
    # Convert to DLR coefficients
    prop_dlr = tau2dlr_propagator(dlr_grid, propagator, dlr_grid.τ)

    # Evaluate at the requested tau
    prop_at_tau = dlr2tau_propagator(dlr_grid, prop_dlr, [tau])

    # Convert single-time-slice propagator to dense matrix
    return propagator_to_matrix(prop_at_tau)
end

"""
    propagator_to_matrix(prop::Propagator{T}) where T

Convert a Propagator with single time slice (n_tau=1) to a dense matrix.
"""
function propagator_to_matrix(prop::Propagator{T}) where {T}
    # Determine total dimension from basis
    n_total = sum(sector.dim for sector in prop.basis.sectors)
    result = zeros(T, n_total, n_total)

    # Fill in blocks
    for (key, block) in prop.blocks
        (i, j) = key
        dim_i = prop.basis.sectors[i].dim
        dim_j = prop.basis.sectors[j].dim

        # Get row and column offsets
        row_offset = sum(prop.basis.sectors[k].dim for k in 1:(i - 1); init = 0)
        col_offset = sum(prop.basis.sectors[k].dim for k in 1:(j - 1); init = 0)

        # Extract single time slice (layout: dim1, dim2, n_tau, so time is at index 3)
        @boundscheck @assert size(block, 3) == 1 "Expected single time slice"
        @inbounds result[(row_offset + 1):(row_offset + dim_i), (col_offset + 1):(col_offset + dim_j)] = block[:, :, 1]
    end

    return result
end

#=============================================================================
# Interface Implementations
=============================================================================#

# DLR transformations - interface versions (dispatch to format-specific implementations)
tau2dlr(dlr_grid::DLRGrid, prop::Propagator{T}, tau_grid::Vector{Float64}) where {T} = tau2dlr_propagator(dlr_grid, prop, tau_grid)

tau2dlr!(dest::Propagator{T}, dlr_grid::DLRGrid,
    prop::Propagator{T}, tau_grid::Vector{Float64}, dlr_workspace::DLRWorkspace{T}) where {T} = tau2dlr_propagator!(dest, dlr_grid, prop, tau_grid, dlr_workspace)

dlr2tau(dlr_grid::DLRGrid, prop_dlr::Propagator{T}, tau_grid::Vector{Float64}) where {T} = dlr2tau_propagator(dlr_grid, prop_dlr, tau_grid)

# Reflection (in-place only, using pre-computed reflection matrix)
do_reflection_tau2tau!(dest::Propagator{T}, prop_tau::Propagator{T},
    reflection_matrix::AbstractMatrix{Float64},
    dlr_workspace::DLRWorkspace{T}) where {T} = do_reflection_tau2tau_propagator!(dest, prop_tau, reflection_matrix, dlr_workspace)

# Extract at tau
extract_at_tau(propagator::Propagator{T}, tau::Float64, dlr_grid::DLRGrid) where {T} = extract_at_tau_propagator(propagator, tau, dlr_grid)

# Convolution
do_convolution_dlr2tau!(dest::Propagator{T}, workspace::ConvolutionWorkspace{T},
    dlr_grid::DLRGrid, kernel_cache::KernelCache,
    F_dlr::Propagator{T}, G_dlr::Propagator{T}) where {T} = do_convolution_dlr2tau_propagator!(dest, workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)

#=============================================================================
# Green's Function DLR Operations
=============================================================================#

"""
    tau2dlr_green(dlr_grid::DLRGrid, arr::AbstractArray{T,3}, tau_grid::Vector{Float64}) where T

Transform Green's function array from tau space to DLR representation (allocating version).
"""
function tau2dlr_green(dlr_grid::DLRGrid, arr::AbstractArray{T, 3}, tau_grid::Vector{Float64}) where {T}
    coeffs = QMBToolkitDLR.tau2dlr(dlr_grid, arr, tau_grid; axis = 3)
    return coeffs
end

"""
    dlr2tau_green(dlr_grid::DLRGrid, arr::AbstractArray{T,3}, tau_grid::Vector{Float64}) where T

Transform Green's function array from DLR representation to tau space.
"""
function dlr2tau_green(dlr_grid::DLRGrid, arr::AbstractArray{T, 3}, tau_grid::Vector{Float64}) where {T}
    vals = QMBToolkitDLR.dlr2tau(dlr_grid, arr, tau_grid; axis = 3)
    return vals
end

# Interface implementations for Green's function arrays
tau2dlr(dlr_grid::DLRGrid, arr::AbstractArray{T, 3}, tau_grid::Vector{Float64}) where {T} = tau2dlr_green(dlr_grid, arr, tau_grid)
dlr2tau(dlr_grid::DLRGrid, arr::AbstractArray{T, 3}, tau_grid::Vector{Float64}) where {T} = dlr2tau_green(dlr_grid, arr, tau_grid)
