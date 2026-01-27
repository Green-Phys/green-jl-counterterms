"""
Convolution operations for dense Array{T,3} representation.

Only the in-place workspace version is provided for performance.
Users should create workspace once and reuse across calls.

Layout: (dim1, dim2, r) where r is DLR rank, for contiguous BLAS access.
"""

using Lehmann
using LinearAlgebra

"""
    do_convolution_dlr2tau!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Array{T,3}, G_dlr::Array{T,3}; do_view::Bool=false) where T<:Number

In-place convolution using pre-allocated workspace buffers.

Computes the convolution H = F * G in DLR representation and returns the result in tau space.
Supports non-square matrix convolutions: F (d1 × d2 × r) * G (d2 × d3 × r) -> H (d1 × d3 × n_tau).

# Arguments
- `workspace`: Pre-allocated workspace buffers
- `dlr_grid`: DLR grid
- `kernel_cache`: Pre-computed kernel cache
- `F_dlr`: First operand (d1 × d2 × r)
- `G_dlr`: Second operand (d2 × d3 × r)
- `do_view`: If `false` (default), workspace dimensions must match F_dlr and G_dlr exactly.
             If `true`, workspace can be larger and views are used for the actual dimensions.
             The view-based mode has ~15-20% overhead but allows a single workspace to be
             reused for different block sizes.

# Returns
- `Array{T,3}`: Result H_tau (d1 × d3 × n_tau)
"""
function do_convolution_dlr2tau!(
    workspace::ConvolutionWorkspace{T},
    dlr_grid::DLRGrid,
    kernel_cache::KernelCache,
    F_dlr::Array{T,3},
    G_dlr::Array{T,3};
    do_view::Bool = false,
) where {T<:Number}
    d1 = size(F_dlr, 1)  # output row dimension
    d2 = size(F_dlr, 2)  # intermediate dimension
    r = size(F_dlr, 3)   # DLR rank
    d3 = size(G_dlr, 2)  # output column dimension

    if do_view
        return _do_convolution_view!(
            workspace,
            dlr_grid,
            kernel_cache,
            F_dlr,
            G_dlr,
            r,
            d1,
            d2,
            d3,
        )
    else
        # Direct mode: workspace dimensions must match exactly
        return _do_convolution_direct!(
            workspace,
            dlr_grid,
            kernel_cache,
            F_dlr,
            G_dlr,
            r,
            d1,
            d2,
            d3,
        )
    end
end

"""
Direct convolution implementation - workspace dimensions match exactly.
Layout: (d1, d3, r) for workspace buffers, (d1, d3, n_tau) for output.
"""
function _do_convolution_direct!(
    workspace::ConvolutionWorkspace{T},
    dlr_grid::DLRGrid,
    kernel_cache::KernelCache,
    F_dlr::Array{T,3},
    G_dlr::Array{T,3},
    r::Int,
    d1::Int,
    d2::Int,
    d3::Int,
) where {T<:Number}
    # Clear workspace buffers
    fill!(workspace.H1_dlr, zero(T))
    fill!(workspace.H2_dlr, zero(T))

    # Compute H1_dlr using workspace - contiguous matrix slices at [:, :, j]
    @inbounds for j = 1:r
        mul!(workspace.temp_FG, @view(F_dlr[:, :, j]), @view(G_dlr[:, :, j]))
        @view(workspace.H1_dlr[:, :, j]) .= kernel_cache.values[j] .* workspace.temp_FG
    end

    # Compute H2_dlr using workspace with in-place operations
    @inbounds for j = 1:r
        fill!(workspace.term1, zero(T))
        fill!(workspace.term2, zero(T))

        # Accumulate terms without broadcasting allocations
        for k = 1:r
            if k != j
                coeff = kernel_cache.values[k] / (dlr_grid.ω[k] - dlr_grid.ω[j])
                workspace.term1 .+= coeff .* @view(F_dlr[:, :, k])
                workspace.term2 .+= coeff .* @view(G_dlr[:, :, k])
            end
        end

        # Multiply accumulated terms by the j-th slice
        mul!(workspace.term1_result, workspace.term1, @view(G_dlr[:, :, j]))
        mul!(workspace.term2_result, @view(F_dlr[:, :, j]), workspace.term2)
        @view(workspace.H2_dlr[:, :, j]) .= workspace.term1_result .+ workspace.term2_result
    end

    # Transform to tau space with axis=3 for new layout
    H1_tau = dlr2tau(dlr_grid, workspace.H1_dlr, dlr_grid.τ; axis = 3)
    H2_tau = dlr2tau(dlr_grid, workspace.H2_dlr, dlr_grid.τ; axis = 3)

    # Compute final result: tau broadcasts along 3rd dimension
    # Reshape tau to (1, 1, n_tau) for broadcasting with (d1, d3, n_tau)
    n_tau = length(dlr_grid.τ)
    tau_reshaped = reshape(dlr_grid.τ, 1, 1, n_tau)
    H_tau = tau_reshaped .* H1_tau .+ H2_tau
    return H_tau
end

"""
View-based convolution implementation - workspace can be larger than needed.
Uses views into workspace buffers based on actual dimensions.
Layout: (d1, d3, r) for workspace buffers, (d1, d3, n_tau) for output.
"""
function _do_convolution_view!(
    workspace::ConvolutionWorkspace{T},
    dlr_grid::DLRGrid,
    kernel_cache::KernelCache,
    F_dlr::Array{T,3},
    G_dlr::Array{T,3},
    r::Int,
    d1::Int,
    d2::Int,
    d3::Int,
) where {T<:Number}
    # Create views into workspace buffers for the actual dimensions needed
    # New layout: (d1, d3, r)
    H1_dlr_view = @view workspace.H1_dlr[1:d1, 1:d3, 1:r]
    H2_dlr_view = @view workspace.H2_dlr[1:d1, 1:d3, 1:r]
    term1_view = @view workspace.term1[1:d1, 1:d2]
    term2_view = @view workspace.term2[1:d2, 1:d3]
    term1_result_view = @view workspace.term1_result[1:d1, 1:d3]
    term2_result_view = @view workspace.term2_result[1:d1, 1:d3]

    # Clear only the portions we use
    fill!(H1_dlr_view, zero(T))
    fill!(H2_dlr_view, zero(T))

    # Compute H1_dlr - contiguous matrix slices at [:, :, j]
    temp_FG_view = @view workspace.temp_FG[1:d1, 1:d3]
    @inbounds for j = 1:r
        mul!(temp_FG_view, @view(F_dlr[:, :, j]), @view(G_dlr[:, :, j]))
        @view(H1_dlr_view[:, :, j]) .= kernel_cache.values[j] .* temp_FG_view
    end

    # Compute H2_dlr
    @inbounds for j = 1:r
        fill!(term1_view, zero(T))
        fill!(term2_view, zero(T))

        for k = 1:r
            if k != j
                coeff = kernel_cache.values[k] / (dlr_grid.ω[k] - dlr_grid.ω[j])
                term1_view .+= coeff .* @view(F_dlr[:, :, k])
                term2_view .+= coeff .* @view(G_dlr[:, :, k])
            end
        end

        mul!(term1_result_view, term1_view, @view(G_dlr[:, :, j]))
        mul!(term2_result_view, @view(F_dlr[:, :, j]), term2_view)
        @view(H2_dlr_view[:, :, j]) .= term1_result_view .+ term2_result_view
    end

    # Transform to tau space with axis=3
    # Need to collect to contiguous arrays as strided views are slower with dlr2tau
    H1_tau = dlr2tau(dlr_grid, collect(H1_dlr_view), dlr_grid.τ; axis = 3)
    H2_tau = dlr2tau(dlr_grid, collect(H2_dlr_view), dlr_grid.τ; axis = 3)

    # Compute final result: tau broadcasts along 3rd dimension
    n_tau = length(dlr_grid.τ)
    tau_reshaped = reshape(dlr_grid.τ, 1, 1, n_tau)
    H_tau = tau_reshaped .* H1_tau .+ H2_tau
    return H_tau
end
