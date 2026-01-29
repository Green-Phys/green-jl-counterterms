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
Uses in-place dlr2tau! for efficient tau-space transformation.
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
    n_tau = length(dlr_grid.τ)
    dlr_size = d1 * d3 * r
    tau_size = d1 * d3 * n_tau

    # Reshape contiguous buffers once at start
    H1_dlr = reshape(@view(workspace.dlr_buffer1[1:dlr_size]), d1, d3, r)
    H2_dlr = reshape(@view(workspace.dlr_buffer2[1:dlr_size]), d1, d3, r)
    H1_tau = reshape(@view(workspace.tau_buffer1[1:tau_size]), d1, d3, n_tau)
    H2_tau = reshape(@view(workspace.tau_buffer2[1:tau_size]), d1, d3, n_tau)
    H_tau = reshape(@view(workspace.tau_result[1:tau_size]), d1, d3, n_tau)

    # Clear workspace buffers
    fill!(H1_dlr, zero(T))
    fill!(H2_dlr, zero(T))

    # Compute H1_dlr - contiguous matrix slices at [:, :, j]
    @inbounds for j = 1:r
        mul!(workspace.temp_FG, @view(F_dlr[:, :, j]), @view(G_dlr[:, :, j]))
        @view(H1_dlr[:, :, j]) .= kernel_cache.values[j] .* workspace.temp_FG
    end

    # Compute H2_dlr using pre-computed convolution coefficients
    coeffs = kernel_cache.convolution_coeffs
    @inbounds for j = 1:r
        fill!(workspace.term1, zero(T))
        fill!(workspace.term2, zero(T))

        # Accumulate terms using axpy! to avoid broadcast temporaries
        for k = 1:r
            if k != j
                coeff = coeffs[k, j]
                axpy!(coeff, @view(F_dlr[:, :, k]), workspace.term1)
                axpy!(coeff, @view(G_dlr[:, :, k]), workspace.term2)
            end
        end

        # Multiply accumulated terms by the j-th slice
        mul!(workspace.term1_result, workspace.term1, @view(G_dlr[:, :, j]))
        mul!(workspace.term2_result, @view(F_dlr[:, :, j]), workspace.term2)
        @view(H2_dlr[:, :, j]) .= workspace.term1_result .+ workspace.term2_result
    end

    # Transform to tau space in-place using dlr2tau!
    dlr2tau!(H1_tau, dlr_grid, H1_dlr, workspace.dlr_workspace)
    dlr2tau!(H2_tau, dlr_grid, H2_dlr, workspace.dlr_workspace)

    # Compute final result in-place: H_tau = tau * H1_tau + H2_tau
    @inbounds for k = 1:n_tau
        tau_k = dlr_grid.τ[k]
        for j = 1:d3
            @simd for i = 1:d1
                H_tau[i, j, k] = tau_k * H1_tau[i, j, k] + H2_tau[i, j, k]
            end
        end
    end
    return H_tau
end

"""
View-based convolution implementation - workspace can be larger than needed.
Uses views into workspace buffers based on actual dimensions.
Layout: (d1, d3, r) for workspace buffers, (d1, d3, n_tau) for output.

Computes directly into contiguous linear buffers to avoid intermediate allocations.
Returns a reshaped view of workspace.tau_result - caller should copy if needed.
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
    n_tau = length(dlr_grid.τ)
    dlr_size = d1 * d3 * r
    tau_size = d1 * d3 * n_tau

    # Reshape contiguous buffers ONCE at start - compute directly into these
    H1_dlr = reshape(@view(workspace.dlr_buffer1[1:dlr_size]), d1, d3, r)
    H2_dlr = reshape(@view(workspace.dlr_buffer2[1:dlr_size]), d1, d3, r)
    H1_tau = reshape(@view(workspace.tau_buffer1[1:tau_size]), d1, d3, n_tau)
    H2_tau = reshape(@view(workspace.tau_buffer2[1:tau_size]), d1, d3, n_tau)
    H_tau = reshape(@view(workspace.tau_result[1:tau_size]), d1, d3, n_tau)

    # Clear only the portions we use
    fill!(H1_dlr, zero(T))
    fill!(H2_dlr, zero(T))

    # Create views for intermediate computation
    term1_view = @view workspace.term1[1:d1, 1:d2]
    term2_view = @view workspace.term2[1:d2, 1:d3]
    term1_result_view = @view workspace.term1_result[1:d1, 1:d3]
    term2_result_view = @view workspace.term2_result[1:d1, 1:d3]
    temp_FG_view = @view workspace.temp_FG[1:d1, 1:d3]

    # Compute H1_dlr - directly into contiguous buffer
    @inbounds for j = 1:r
        mul!(temp_FG_view, @view(F_dlr[:, :, j]), @view(G_dlr[:, :, j]))
        @view(H1_dlr[:, :, j]) .= kernel_cache.values[j] .* temp_FG_view
    end

    # Compute H2_dlr using pre-computed convolution coefficients - directly into contiguous buffer
    coeffs = kernel_cache.convolution_coeffs
    @inbounds for j = 1:r
        fill!(term1_view, zero(T))
        fill!(term2_view, zero(T))

        # Accumulate terms using axpy! to avoid broadcast temporaries
        for k = 1:r
            if k != j
                coeff = coeffs[k, j]
                axpy!(coeff, @view(F_dlr[:, :, k]), term1_view)
                axpy!(coeff, @view(G_dlr[:, :, k]), term2_view)
            end
        end

        mul!(term1_result_view, term1_view, @view(G_dlr[:, :, j]))
        mul!(term2_result_view, @view(F_dlr[:, :, j]), term2_view)
        @view(H2_dlr[:, :, j]) .= term1_result_view .+ term2_result_view
    end

    # Transform directly from contiguous buffers to tau space
    dlr2tau!(H1_tau, dlr_grid, H1_dlr, workspace.dlr_workspace)
    dlr2tau!(H2_tau, dlr_grid, H2_dlr, workspace.dlr_workspace)

    # Compute final result in-place
    @inbounds for k = 1:n_tau
        tau_k = dlr_grid.τ[k]
        for j = 1:d3
            @simd for i = 1:d1
                H_tau[i, j, k] = tau_k * H1_tau[i, j, k] + H2_tau[i, j, k]
            end
        end
    end
    return H_tau
end
