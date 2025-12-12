"""
Convolution operations for dense Array{T,3} representation.

Only the in-place workspace version is provided for performance.
Users should create workspace once and reuse across calls.
"""

using Lehmann

"""
    do_convolution_dlr2tau!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Array{T,3}, G_dlr::Array{T,3}) where T<:Number

In-place version using pre-allocated workspace buffers.
Eliminates ~1.4 GB of allocations per call by reusing workspace.

This is the most optimized version. Create workspace once and reuse across calls.
"""
function do_convolution_dlr2tau!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Array{T,3}, G_dlr::Array{T,3}) where T<:Number
    r = size(F_dlr, 1)
    o1 = size(F_dlr, 2)
    o2 = size(F_dlr, 3)

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
                workspace.term1 .+= coeff .* @view(F_dlr[k, :, :])
                workspace.term2 .+= coeff .* @view(G_dlr[k, :, :])
            end
        end

        workspace.term1 .= workspace.term1 * @view(G_dlr[j, :, :])
        workspace.term2 .= @view(F_dlr[j, :, :]) * workspace.term2
        @view(workspace.H2_dlr[j, :, :]) .= workspace.term1 .+ workspace.term2
    end

    # Transform to tau space (dlr2tau writes to provided buffers if we could modify it, but for now allocates)
    workspace.H1_tau .= dlr2tau(dlr_grid, workspace.H1_dlr, dlr_grid.tau)
    workspace.H2_tau .= dlr2tau(dlr_grid, workspace.H2_dlr, dlr_grid.tau)

    # Compute final result
    H_tau = dlr_grid.tau .* workspace.H1_tau .+ workspace.H2_tau
    return H_tau
end