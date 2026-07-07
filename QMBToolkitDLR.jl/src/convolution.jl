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

    # Fused τ-weighted DLR to tau transformation
    # H_tau = τ * (kernel * H1_dlr) + kernel * H2_dlr
    #       = tau_weighted_kernel * H1_dlr + kernel * H2_dlr
    # This saves one intermediate copy and the τ multiplication loop

    # Copy H1_dlr to contiguous buffer (r, d1*d3)
    H1_matrix = view(workspace.dlr_workspace.buffer1, 1:r, 1:(d1*d3))
    Lehmann._tensor3_to_matrix!(H1_matrix, H1_dlr)

    # Copy H2_dlr to contiguous buffer (r, d1*d3)
    H2_matrix = view(workspace.dlr_workspace.buffer2, 1:r, 1:(d1*d3))
    Lehmann._tensor3_to_matrix!(H2_matrix, H2_dlr)

    # Compute H_tau in matrix form using tau_buffer1 as intermediate buffer
    # (to avoid aliasing with tau_result which is used for tensor output)
    H_tau_matrix = reshape(@view(workspace.tau_buffer1[1:tau_size]), n_tau, d1*d3)
    mul!(H_tau_matrix, workspace.tau_weighted_kernel, H1_matrix)  # H_tau = tau_weighted * H1
    mul!(H_tau_matrix, workspace.kernel_tau, H2_matrix, 1.0, 1.0) # H_tau += kernel * H2

    # Copy result back to tensor form
    Lehmann._matrix_to_tensor3!(H_tau, H_tau_matrix)

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

    # Fused τ-weighted DLR to tau transformation
    # H_tau = τ * (kernel * H1_dlr) + kernel * H2_dlr
    #       = tau_weighted_kernel * H1_dlr + kernel * H2_dlr
    # This saves one intermediate copy and the τ multiplication loop

    # Copy H1_dlr to contiguous buffer (r, d1*d3)
    H1_matrix = view(workspace.dlr_workspace.buffer1, 1:r, 1:(d1*d3))
    Lehmann._tensor3_to_matrix!(H1_matrix, H1_dlr)

    # Copy H2_dlr to contiguous buffer (r, d1*d3)
    H2_matrix = view(workspace.dlr_workspace.buffer2, 1:r, 1:(d1*d3))
    Lehmann._tensor3_to_matrix!(H2_matrix, H2_dlr)

    # Compute H_tau in matrix form using tau_buffer1 as intermediate buffer
    # (to avoid aliasing with tau_result which is used for tensor output)
    H_tau_matrix = reshape(@view(workspace.tau_buffer1[1:tau_size]), n_tau, d1*d3)
    mul!(H_tau_matrix, workspace.tau_weighted_kernel, H1_matrix)  # H_tau = tau_weighted * H1
    mul!(H_tau_matrix, workspace.kernel_tau, H2_matrix, 1.0, 1.0) # H_tau += kernel * H2

    # Copy result back to tensor form
    Lehmann._matrix_to_tensor3!(H_tau, H_tau_matrix)

    return H_tau
end

"""
Shared fused τ-weighted DLR→tau transform for the diagonal-operand convolution paths.
Computes H_tau = tau_weighted_kernel * H1_dlr + kernel_tau * H2_dlr, identical to the
tail of `_do_convolution_view!`. H1_dlr/H2_dlr are (d1 × d3 × r), H_tau is (d1 × d3 × n_tau).
"""
@inline function _fused_dlr2tau!(
    workspace::ConvolutionWorkspace{T},
    H1_dlr,
    H2_dlr,
    H_tau,
    d1::Int,
    d3::Int,
    r::Int,
    n_tau::Int,
) where {T<:Number}
    tau_size = d1 * d3 * n_tau

    H1_matrix = view(workspace.dlr_workspace.buffer1, 1:r, 1:(d1*d3))
    Lehmann._tensor3_to_matrix!(H1_matrix, H1_dlr)

    H2_matrix = view(workspace.dlr_workspace.buffer2, 1:r, 1:(d1*d3))
    Lehmann._tensor3_to_matrix!(H2_matrix, H2_dlr)

    H_tau_matrix = reshape(@view(workspace.tau_buffer1[1:tau_size]), n_tau, d1*d3)
    mul!(H_tau_matrix, workspace.tau_weighted_kernel, H1_matrix)  # H_tau = tau_weighted * H1
    mul!(H_tau_matrix, workspace.kernel_tau, H2_matrix, 1.0, 1.0) # H_tau += kernel * H2

    Lehmann._matrix_to_tensor3!(H_tau, H_tau_matrix)
    return H_tau
end

"""
    do_convolution_dlr2tau_Fdiag!(workspace, dlr_grid, kernel_cache, Fdiag, G_dlr; do_view=true)

Convolution H = F * G where the left operand F is diagonal within-block, supplied as its
diagonal `Fdiag` (d × r) with F[:, :, j] = Diagonal(Fdiag[:, j]). G_dlr is dense (d × d3 × r).
Returns H_tau (d × d3 × n_tau) as a view of `workspace.tau_result`.

Because F is diagonal, every per-slice matmul `F[:,:,j] * (...)` collapses to a row scaling,
removing the O(d) contraction dimension from the convolution's dominant matmuls.
"""
function do_convolution_dlr2tau_Fdiag!(
    workspace::ConvolutionWorkspace{T},
    dlr_grid::DLRGrid,
    kernel_cache::KernelCache,
    Fdiag::AbstractMatrix{T},
    G_dlr::Array{T,3};
    do_view::Bool = true,
) where {T<:Number}
    d = size(Fdiag, 1)
    r = size(Fdiag, 2)
    d3 = size(G_dlr, 2)
    n_tau = length(dlr_grid.τ)
    dlr_size = d * d3 * r
    tau_size = d * d3 * n_tau

    H1_dlr = reshape(@view(workspace.dlr_buffer1[1:dlr_size]), d, d3, r)
    H2_dlr = reshape(@view(workspace.dlr_buffer2[1:dlr_size]), d, d3, r)
    H_tau = reshape(@view(workspace.tau_result[1:tau_size]), d, d3, n_tau)
    fill!(H1_dlr, zero(T))
    fill!(H2_dlr, zero(T))

    term2_view = @view workspace.term2[1:d, 1:d3]   # dense G accumulation
    t1col = @view workspace.temp_FG[1:d, 1]          # diagonal F accumulation (scratch)
    coeffs = kernel_cache.convolution_coeffs

    # H1[:, :, j] = kernel[j] * Diagonal(Fdiag[:, j]) * G[:, :, j]  (row scaling)
    @inbounds for j = 1:r
        kv = kernel_cache.values[j]
        @view(H1_dlr[:, :, j]) .= (kv .* @view(Fdiag[:, j])) .* @view(G_dlr[:, :, j])
    end

    # H2[:, :, j] = t1col .* G[:, :, j] + Fdiag[:, j] .* term2
    #   t1col[a] = Σ_{k≠j} coeffs[k,j] Fdiag[a,k]     (diagonal, so a vector)
    #   term2    = Σ_{k≠j} coeffs[k,j] G[:, :, k]     (dense)
    @inbounds for j = 1:r
        fill!(t1col, zero(T))
        fill!(term2_view, zero(T))
        for k = 1:r
            if k != j
                c = coeffs[k, j]
                axpy!(c, @view(Fdiag[:, k]), t1col)
                axpy!(c, @view(G_dlr[:, :, k]), term2_view)
            end
        end
        @view(H2_dlr[:, :, j]) .=
            t1col .* @view(G_dlr[:, :, j]) .+ @view(Fdiag[:, j]) .* term2_view
    end

    return _fused_dlr2tau!(workspace, H1_dlr, H2_dlr, H_tau, d, d3, r, n_tau)
end

"""
    do_convolution_dlr2tau_Gdiag!(workspace, dlr_grid, kernel_cache, F_dlr, Gdiag; do_view=true)

Convolution H = F * G where the right operand G is diagonal within-block, supplied as its
diagonal `Gdiag` (d × r) with G[:, :, j] = Diagonal(Gdiag[:, j]). F_dlr is dense (d1 × d × r).
Returns H_tau (d1 × d × n_tau) as a view of `workspace.tau_result`.

Because G is diagonal, every per-slice matmul `(...) * G[:,:,j]` collapses to a column scaling.
"""
function do_convolution_dlr2tau_Gdiag!(
    workspace::ConvolutionWorkspace{T},
    dlr_grid::DLRGrid,
    kernel_cache::KernelCache,
    F_dlr::Array{T,3},
    Gdiag::AbstractMatrix{T};
    do_view::Bool = true,
) where {T<:Number}
    d1 = size(F_dlr, 1)
    d = size(Gdiag, 1)   # intermediate = output column dimension
    r = size(Gdiag, 2)
    n_tau = length(dlr_grid.τ)
    dlr_size = d1 * d * r
    tau_size = d1 * d * n_tau

    H1_dlr = reshape(@view(workspace.dlr_buffer1[1:dlr_size]), d1, d, r)
    H2_dlr = reshape(@view(workspace.dlr_buffer2[1:dlr_size]), d1, d, r)
    H_tau = reshape(@view(workspace.tau_result[1:tau_size]), d1, d, n_tau)
    fill!(H1_dlr, zero(T))
    fill!(H2_dlr, zero(T))

    term1_view = @view workspace.term1[1:d1, 1:d]   # dense F accumulation
    t2col = @view workspace.temp_FG[1:d, 1]          # diagonal G accumulation (scratch)
    coeffs = kernel_cache.convolution_coeffs

    # H1[:, :, j] = kernel[j] * F[:, :, j] * Diagonal(Gdiag[:, j])  (column scaling)
    @inbounds for j = 1:r
        kv = kernel_cache.values[j]
        @view(H1_dlr[:, :, j]) .=
            (kv .* @view(F_dlr[:, :, j])) .* transpose(@view(Gdiag[:, j]))
    end

    # H2[:, :, j] = term1 .* Gdiag[:, j]' + F[:, :, j] .* t2col'
    #   term1    = Σ_{k≠j} coeffs[k,j] F[:, :, k]     (dense)
    #   t2col[c] = Σ_{k≠j} coeffs[k,j] Gdiag[c,k]     (diagonal, so a vector)
    @inbounds for j = 1:r
        fill!(term1_view, zero(T))
        fill!(t2col, zero(T))
        for k = 1:r
            if k != j
                c = coeffs[k, j]
                axpy!(c, @view(F_dlr[:, :, k]), term1_view)
                axpy!(c, @view(Gdiag[:, k]), t2col)
            end
        end
        @view(H2_dlr[:, :, j]) .=
            term1_view .* transpose(@view(Gdiag[:, j])) .+
            @view(F_dlr[:, :, j]) .* transpose(t2col)
    end

    return _fused_dlr2tau!(workspace, H1_dlr, H2_dlr, H_tau, d1, d, r, n_tau)
end
