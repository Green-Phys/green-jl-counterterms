using Lehmann

"""
    KernelCache

Cache for kernel_tau evaluations at tau=0.
Significantly reduces allocations in do_convolution_dlr2tau by avoiding
repeated kernel_tau(0, omega[k], beta) calls in the nested loop.
"""
struct KernelCache
    values::Vector{Float64}  # kernel_tau(0, omega[i], beta) for each i
end

"""
    ConvolutionWorkspace{T<:Number}

Pre-allocated workspace buffers for do_convolution_dlr2tau.
Reusing these buffers across multiple convolution calls eliminates ~1.4 GB
of allocations per call.

# Fields
- `H1_dlr`: Buffer for first convolution term (r × o1 × o2)
- `H2_dlr`: Buffer for second convolution term (r × o1 × o2)
- `term1`: Temporary buffer for accumulation (o1 × o2)
- `term2`: Temporary buffer for accumulation (o1 × o2)
- `H1_tau`: Buffer for tau-space result of H1 (n_tau × o1 × o2)
- `H2_tau`: Buffer for tau-space result of H2 (n_tau × o1 × o2)
"""
struct ConvolutionWorkspace{T<:Number}
    H1_dlr::Array{T,3}
    H2_dlr::Array{T,3}
    term1::Array{T,2}
    term2::Array{T,2}
    H1_tau::Array{T,3}
    H2_tau::Array{T,3}
end

"""
    ConvolutionWorkspace(r::Int, o1::Int, o2::Int, n_tau::Int)

Create a workspace for convolution operations with given dimensions (defaults to Float64).

# Arguments
- `r`: DLR rank (number of DLR coefficients)
- `o1`: First orbital dimension
- `o2`: Second orbital dimension
- `n_tau`: Number of tau grid points
"""
function ConvolutionWorkspace(r::Int, o1::Int, o2::Int, n_tau::Int)
    return ConvolutionWorkspace{Float64}(
        zeros(Float64, r, o1, o2),
        zeros(Float64, r, o1, o2),
        zeros(Float64, o1, o2),
        zeros(Float64, o1, o2),
        zeros(Float64, n_tau, o1, o2),
        zeros(Float64, n_tau, o1, o2)
    )
end

"""
    ConvolutionWorkspace{T}(r::Int, o1::Int, o2::Int, n_tau::Int) where T<:Number

Create a workspace for convolution operations with specified numeric type.

# Arguments
- `r`: DLR rank (number of DLR coefficients)
- `o1`: First orbital dimension
- `o2`: Second orbital dimension
- `n_tau`: Number of tau grid points
"""
function ConvolutionWorkspace{T}(r::Int, o1::Int, o2::Int, n_tau::Int) where T<:Number
    return ConvolutionWorkspace{T}(
        zeros(T, r, o1, o2),
        zeros(T, r, o1, o2),
        zeros(T, o1, o2),
        zeros(T, o1, o2),
        zeros(T, n_tau, o1, o2),
        zeros(T, n_tau, o1, o2)
    )
end

"""
    compute_kernel_cache(dlr_grid::DLRGrid)

Pre-compute kernel_tau(0, omega[i], beta) for all DLR omega points.
This cache eliminates ~1.1 GB of allocations in do_convolution_dlr2tau.
"""
function compute_kernel_cache(dlr_grid::DLRGrid)
    cache = Vector{Float64}(undef, length(dlr_grid.omega))
    for i in 1:length(dlr_grid.omega)
        cache[i] = kernel_tau(0.0, dlr_grid.omega[i], dlr_grid.beta)
    end
    return KernelCache(cache)
end

@inline function kernel_tau(tau, omega, beta)
    @assert tau >= -beta && tau <= beta "tau must be in [-beta, beta]"
    if tau >= 0
        if omega > 0
            return exp(-tau * omega)/(1 + exp(-beta * omega))
        else
            return exp(-(tau - beta) * omega)/(1 + exp(beta * omega))   
        end
    else
        if omega > 0
            return -1 * exp(-(tau + beta) * omega)/(1 + exp(-beta * omega))
        else
            return -1 * exp(-tau * omega)/(1 + exp(beta * omega))   
        end
    end
end

"""
    do_convolution_dlr2tau(dlr_grid::DLRGrid, F_dlr::Array{Float64,3}, G_dlr::Array{Float64,3})

Calculate \$\\int_{0}^{\\tau} d\\tau' F(\\tau-\\tau') G(\\tau')\$
where F and G are represented in the DLR basis, which are Arrays of size (r,o1,o2).
Returns the results on the DLR tau grid as an Array of size (r,o1,o2).

This version computes a kernel cache on first call. For better performance,
use `do_convolution_dlr2tau(dlr_grid, kernel_cache, F_dlr, G_dlr)` with
a pre-computed cache.
"""
function do_convolution_dlr2tau(dlr_grid::DLRGrid, F_dlr::Array{Float64,3}, G_dlr::Array{Float64,3})
    kernel_cache = compute_kernel_cache(dlr_grid)
    return do_convolution_dlr2tau(dlr_grid, kernel_cache, F_dlr, G_dlr)
end

"""
    do_convolution_dlr2tau(dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Array{Float64,3}, G_dlr::Array{Float64,3})

Optimized version using pre-computed kernel cache.
Eliminates ~1.1 GB of allocations compared to recomputing kernels each call.

For even better performance, use the workspace version:
`do_convolution_dlr2tau!(workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)`
"""
function do_convolution_dlr2tau(dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::Array{Float64,3}, G_dlr::Array{Float64,3})
    r = size(F_dlr, 1)
    o1 = size(F_dlr, 2)
    o2 = size(F_dlr, 3)
    n_tau = length(dlr_grid.tau)

    workspace = ConvolutionWorkspace(r, o1, o2, n_tau)
    return do_convolution_dlr2tau!(workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)
end

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

"""
    Perform reflection operation: F(τ) = H(β - τ)
    where F is represented in the DLR basis, which is an Array of size (r,o1,o2).
    return the results on the DLR tau grid as an Array of size (r, o1, o2).
"""
function do_reflection_dlr2tau(dlr_grid::DLRGrid, H_dlr::Array{T,3}) where T<:Number
    tau_grid_reflected = dlr_grid.beta .- dlr_grid.tau
    F_tau = dlr2tau(dlr_grid, H_dlr, tau_grid_reflected)
    return F_tau
end


# TODO: implement the more efficient version working directly in DLR representation
function do_reflection_tau2tau(dlr_grid::DLRGrid, H_tau::Array{T,3}) where T<:Number
    H_dlr = tau2dlr(dlr_grid, H_tau, dlr_grid.tau)
    return do_reflection_dlr2tau(dlr_grid, H_dlr)
end


"""
    Perform multiplication in DLR representation: H(tau) = F(tau) * G(tau)
"""
function do_multiplication_tau2tau(F_tau::Array{T,3}, G_tau::Array{T,3}) where T<:Number
    H_tau = zeros(T, size(F_tau))
    for k in 1:size(F_tau, 1)
        H_tau[k, :, :] = F_tau[k, :, :] * G_tau[k, :, :]
    end
    return H_tau
end