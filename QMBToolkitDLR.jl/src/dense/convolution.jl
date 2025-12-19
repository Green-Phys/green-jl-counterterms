"""
Convolution operations for dense Array{T,3} representation.

Only the in-place workspace version is provided for performance.
Users should create workspace once and reuse across calls.
"""

using Lehmann
using LinearAlgebra

"""
    do_convolution_dlr2tau!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Array{T,3}, G_dlr::Array{T,3}) where T<:Number

In-place version using pre-allocated workspace buffers.
Eliminates ~1.4 GB of allocations per call by reusing workspace.

Supports non-square matrix convolutions: F (r × d1 × d2) * G (r × d2 × d3) -> H (n_tau × d1 × d3).

This is the most optimized version. Create workspace once and reuse across calls.
"""
function do_convolution_dlr2tau!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Array{T,3}, G_dlr::Array{T,3}) where T<:Number
    r = size(F_dlr, 1)
    d1 = size(F_dlr, 2)  # output row dimension
    d2 = size(F_dlr, 3)  # intermediate dimension
    d3 = size(G_dlr, 3)  # output column dimension

    @assert size(G_dlr, 1) == r "F and G must have same DLR rank"
    @assert size(G_dlr, 2) == d2 "Intermediate dimensions must match: F has $d2, G has $(size(G_dlr, 2))"

    # Clear workspace buffers
    fill!(workspace.H1_dlr, zero(T))
    fill!(workspace.H2_dlr, zero(T))

    # Compute H1_dlr using workspace
    @inbounds for j in 1:r
        @view(workspace.H1_dlr[j, :, :]) .= kernel_cache.values[j] .* (@view(F_dlr[j, :, :]) * @view(G_dlr[j, :, :]))
    end

    # Compute H2_dlr using workspace with in-place operations
    @inbounds for j in 1:r
        fill!(workspace.term1, zero(T))
        fill!(workspace.term2, zero(T))

        # Accumulate terms without broadcasting allocations
        for k in 1:r
            if k != j
                coeff = kernel_cache.values[k] / (dlr_grid.omega[k] - dlr_grid.omega[j])
                workspace.term1 .+= coeff .* @view(F_dlr[k, :, :])  # (d1, d2) += (d1, d2)
                workspace.term2 .+= coeff .* @view(G_dlr[k, :, :])  # (d2, d3) += (d2, d3)
            end
        end

        # Multiply accumulated terms by the j-th slice
        mul!(workspace.term1_result, workspace.term1, @view(G_dlr[j, :, :]))  # (d1, d3) = (d1, d2) * (d2, d3)
        mul!(workspace.term2_result, @view(F_dlr[j, :, :]), workspace.term2)  # (d1, d3) = (d1, d2) * (d2, d3)
        @view(workspace.H2_dlr[j, :, :]) .= workspace.term1_result .+ workspace.term2_result
    end

    # Transform to tau space (dlr2tau writes to provided buffers if we could modify it, but for now allocates)
    workspace.H1_tau .= dlr2tau(dlr_grid, workspace.H1_dlr, dlr_grid.tau)
    workspace.H2_tau .= dlr2tau(dlr_grid, workspace.H2_dlr, dlr_grid.tau)

    # Compute final result
    H_tau = dlr_grid.tau .* workspace.H1_tau .+ workspace.H2_tau
    return H_tau
end