"""
Workspace structures and utilities for DLR operations.

These are format-independent and used by both dense and blockdense implementations.
"""

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

"""
    kernel_tau(tau, omega, beta)

Compute the DLR kernel K(tau, omega) at given tau, omega, and beta.
"""
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