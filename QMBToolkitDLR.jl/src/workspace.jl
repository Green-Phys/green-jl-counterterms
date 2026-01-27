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
Reusing these buffers across multiple convolution calls reduces allocations
in the main computation loop.

Layout: (d1, d2, r) where r is DLR rank, matching the propagator layout (dim1, dim2, n_tau).
Supports non-square matrix convolutions: F (d1 × d2 × r) * G (d2 × d3 × r) -> H (d1 × d3 × n_tau)

# Fields
- `H1_dlr`: Buffer for first convolution term (d1 × d3 × r)
- `H2_dlr`: Buffer for second convolution term (d1 × d3 × r)
- `term1`: Temporary buffer for F accumulation (d1 × d2)
- `term2`: Temporary buffer for G accumulation (d2 × d3)
- `term1_result`: Result buffer for term1 * G (d1 × d3)
- `term2_result`: Result buffer for F * term2 (d1 × d3)
- `temp_FG`: Temporary buffer for F[j] * G[j] matrix multiplication (d1 × d3)
"""
struct ConvolutionWorkspace{T<:Number}
    H1_dlr::Array{T,3}
    H2_dlr::Array{T,3}
    term1::Array{T,2}
    term2::Array{T,2}
    term1_result::Array{T,2}
    term2_result::Array{T,2}
    temp_FG::Array{T,2}
end

"""
    ConvolutionWorkspace(r::Int, o1::Int, o2::Int, n_tau::Int)

Create a workspace for square matrix convolution operations (defaults to Float64).
Layout: (dim1, dim2, r) for contiguous BLAS access.
For F (o1 × o2 × r) * G (o2 × o2 × r) -> H (o1 × o2 × n_tau).

# Arguments
- `r`: DLR rank (number of DLR coefficients)
- `o1`: First orbital dimension
- `o2`: Second orbital dimension (same as intermediate and output dimension)
- `n_tau`: Number of tau grid points
"""
function ConvolutionWorkspace(r::Int, o1::Int, o2::Int, n_tau::Int)
    return ConvolutionWorkspace{Float64}(r, o1, o2, o2, n_tau)
end

"""
    ConvolutionWorkspace{T}(r::Int, d1::Int, d2::Int, d3::Int, n_tau::Int) where T<:Number

Create a workspace for non-square matrix convolution with specified numeric type.
Layout: (dim1, dim2, r) for contiguous BLAS access.
For F (d1 × d2 × r) * G (d2 × d3 × r) -> H (d1 × d3 × n_tau).

# Arguments
- `r`: DLR rank (number of DLR coefficients)
- `d1`: Output row dimension (target dimension)
- `d2`: Intermediate dimension
- `d3`: Output column dimension (source dimension)
- `n_tau`: Number of tau grid points
"""
function ConvolutionWorkspace{T}(
    r::Int,
    d1::Int,
    d2::Int,
    d3::Int,
    n_tau::Int,
) where {T<:Number}
    # n_tau is kept for API compatibility but not used for workspace allocation
    # Layout: (d1, d3, r) for contiguous matrix slices
    return ConvolutionWorkspace{T}(
        zeros(T, d1, d3, r),        # H1_dlr
        zeros(T, d1, d3, r),        # H2_dlr
        zeros(T, d1, d2),           # term1 (accumulates F slices)
        zeros(T, d2, d3),           # term2 (accumulates G slices)
        zeros(T, d1, d3),           # term1_result (term1 * G)
        zeros(T, d1, d3),           # term2_result (F * term2)
        zeros(T, d1, d3),           # temp_FG (F[j] * G[j] result)
    )
end

"""
    ConvolutionWorkspace{T}(r::Int, o1::Int, o2::Int, n_tau::Int) where T<:Number

Create a workspace for square matrix convolution with specified numeric type.
Layout: (dim1, dim2, r) for contiguous BLAS access.
For F (o1 × o2 × r) * G (o2 × o2 × r) -> H (o1 × o2 × n_tau).

# Arguments
- `r`: DLR rank (number of DLR coefficients)
- `o1`: First orbital dimension
- `o2`: Second orbital dimension (same as intermediate and output dimension)
- `n_tau`: Number of tau grid points
"""
function ConvolutionWorkspace{T}(r::Int, o1::Int, o2::Int, n_tau::Int) where {T<:Number}
    return ConvolutionWorkspace{T}(r, o1, o2, o2, n_tau)
end

"""
    compute_kernel_cache(dlr_grid::DLRGrid)

Pre-compute kernel_tau(0, omega[i], beta) for all DLR omega points.
This cache eliminates ~1.1 GB of allocations in do_convolution_dlr2tau.
"""
function compute_kernel_cache(dlr_grid::DLRGrid)
    cache = Vector{Float64}(undef, length(dlr_grid.ω))
    for i = 1:length(dlr_grid.ω)
        cache[i] = kernel_tau(0.0, dlr_grid.ω[i], dlr_grid.β)
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
