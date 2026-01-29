"""
Workspace structures and utilities for DLR operations.

These are format-independent and used by both dense and blockdense implementations.
"""

using Lehmann

"""
    KernelCache

Cache for kernel_tau evaluations and derived coefficients.
Significantly reduces allocations and computation in do_convolution_dlr2tau.

# Fields
- `values`: kernel_tau(0, omega[i], beta) for each DLR pole i
- `convolution_coeffs`: Pre-computed matrix where coeffs[k,j] = values[k] / (ω[k] - ω[j])
  for k ≠ j, and coeffs[k,k] = 0 (diagonal unused but set to 0 for safety)
"""
struct KernelCache
    values::Vector{Float64}  # kernel_tau(0, omega[i], beta) for each i
    convolution_coeffs::Matrix{Float64}  # values[k] / (ω[k] - ω[j]) for k ≠ j
end

"""
    ConvolutionWorkspace{T<:Number}

Pre-allocated workspace buffers for do_convolution_dlr2tau.
Reusing these buffers across multiple convolution calls reduces allocations
in the main computation loop.

Layout: (d1, d2, r) where r is DLR rank, matching the propagator layout (dim1, dim2, n_tau).
Supports non-square matrix convolutions: F (d1 × d2 × r) * G (d2 × d3 × r) -> H (d1 × d3 × n_tau)

# Fields
- `term1`: Temporary buffer for F accumulation (d1 × d2)
- `term2`: Temporary buffer for G accumulation (d2 × d3)
- `term1_result`: Result buffer for term1 * G (d1 × d3)
- `term2_result`: Result buffer for F * term2 (d1 × d3)
- `temp_FG`: Temporary buffer for F[j] * G[j] matrix multiplication (d1 × d3)
- `dlr_workspace`: Workspace for in-place DLR transformations
- `dlr_buffer1`, `dlr_buffer2`: Contiguous linear buffers for DLR data (can be reshaped)
- `tau_buffer1`, `tau_result`: Contiguous linear buffers for tau data (can be reshaped)
- `reflection_matrix`: Pre-computed direct reflection matrix R = K(β-τ,ω) * inv(K(τ,ω))
- `kernel_tau`: Pre-computed DLR kernel K(τ,ω) for dlr2tau transformation
- `tau_weighted_kernel`: Pre-computed τ-weighted kernel for fused convolution: diag(τ) * K(τ,ω)
"""
struct ConvolutionWorkspace{T<:Number}
    term1::Array{T,2}
    term2::Array{T,2}
    term1_result::Array{T,2}
    term2_result::Array{T,2}
    temp_FG::Array{T,2}
    dlr_workspace::DLRWorkspace{T}
    # Contiguous linear buffers (can be reshaped for any dimensions up to max)
    dlr_buffer1::Vector{T}
    dlr_buffer2::Vector{T}
    tau_buffer1::Vector{T}
    tau_result::Vector{T}
    # Pre-computed matrices for efficient DLR operations
    reflection_matrix::Matrix{Float64}  # R = K(β-τ,ω) * inv(K(τ,ω))
    kernel_tau::Matrix{Float64}         # K(τ,ω) for dlr2tau
    tau_weighted_kernel::Matrix{Float64} # diag(τ) * K(τ,ω) for fused τ-weighting in convolution
end

"""
    ConvolutionWorkspace(dlr_grid::DLRGrid, o1::Int, o2::Int)

Create a workspace for square matrix convolution operations (defaults to Float64).
Layout: (dim1, dim2, r) for contiguous BLAS access.
For F (o1 × o2 × r) * G (o2 × o2 × r) -> H (o1 × o2 × n_tau).

# Arguments
- `dlr_grid`: DLR grid (provides r, n_tau, and τ for reflection)
- `o1`: First orbital dimension
- `o2`: Second orbital dimension (same as intermediate and output dimension)
"""
function ConvolutionWorkspace(dlr_grid::DLRGrid, o1::Int, o2::Int)
    return ConvolutionWorkspace{Float64}(dlr_grid, o1, o2, o2)
end

"""
    ConvolutionWorkspace{T}(dlr_grid::DLRGrid, d1::Int, d2::Int, d3::Int) where T<:Number

Create a workspace for non-square matrix convolution with specified numeric type.
Layout: (dim1, dim2, r) for contiguous BLAS access.
For F (d1 × d2 × r) * G (d2 × d3 × r) -> H (d1 × d3 × n_tau).

# Arguments
- `dlr_grid`: DLR grid (provides r, n_tau, and τ for reflection)
- `d1`: Output row dimension (target dimension)
- `d2`: Intermediate dimension
- `d3`: Output column dimension (source dimension)
"""
function ConvolutionWorkspace{T}(
    dlr_grid::DLRGrid,
    d1::Int,
    d2::Int,
    d3::Int,
) where {T<:Number}
    r = length(dlr_grid.ω)
    n_tau = length(dlr_grid.τ)
    max_dlr_size = d1 * d3 * r
    max_tau_size = d1 * d3 * n_tau

    # Pre-compute the direct reflection matrix: R = K(β-τ, ω) * inv(K(τ, ω))
    # This allows reflection in a single matrix multiply instead of tau2dlr + dlr2tau
    τ_reflected = dlr_grid.β .- dlr_grid.τ
    kernel_tau_reflected = Spectral.kernelT(
        Float64,
        Val(dlr_grid.isFermi),
        Val(dlr_grid.symmetry),
        τ_reflected,
        dlr_grid.ω,
        dlr_grid.β,
        true,
    )
    # Ensure kernel_τ is computed
    kernel_tau = Spectral.kernelT(
        Float64,
        Val(dlr_grid.isFermi),
        Val(dlr_grid.symmetry),
        dlr_grid.τ,
        dlr_grid.ω,
        dlr_grid.β,
        true,
    )
    # R = K_reflected * inv(K_tau), computed as K_reflected / K_tau
    reflection_matrix = kernel_tau_reflected / kernel_tau

    # Pre-compute τ-weighted kernel for fused convolution: diag(τ) * K(τ,ω)
    # This allows H_tau = tau_weighted_kernel * H1_dlr + kernel_tau * H2_dlr
    # instead of H_tau = τ * (kernel_tau * H1_dlr) + kernel_tau * H2_dlr
    tau_weighted_kernel = similar(kernel_tau)
    @inbounds for j = 1:r
        @simd for k = 1:n_tau
            tau_weighted_kernel[k, j] = dlr_grid.τ[k] * kernel_tau[k, j]
        end
    end

    return ConvolutionWorkspace{T}(
        zeros(T, d1, d2),           # term1 (accumulates F slices)
        zeros(T, d2, d3),           # term2 (accumulates G slices)
        zeros(T, d1, d3),           # term1_result (term1 * G)
        zeros(T, d1, d3),           # term2_result (F * term2)
        zeros(T, d1, d3),           # temp_FG (F[j] * G[j] result)
        DLRWorkspace(T, r, d1 * d3),  # dlr_workspace for in-place transforms
        # Contiguous linear buffers (can be reshaped for any dimensions up to max)
        zeros(T, max_dlr_size),     # dlr_buffer1
        zeros(T, max_dlr_size),     # dlr_buffer2
        zeros(T, max_tau_size),     # tau_buffer1
        zeros(T, max_tau_size),     # tau_result
        reflection_matrix,          # Pre-computed direct reflection matrix
        kernel_tau,                 # Pre-computed DLR kernel
        tau_weighted_kernel,        # Pre-computed τ-weighted kernel
    )
end

"""
    ConvolutionWorkspace{T}(dlr_grid::DLRGrid, o1::Int, o2::Int) where T<:Number

Create a workspace for square matrix convolution with specified numeric type.
Layout: (dim1, dim2, r) for contiguous BLAS access.
For F (o1 × o2 × r) * G (o2 × o2 × r) -> H (o1 × o2 × n_tau).

# Arguments
- `dlr_grid`: DLR grid (provides r, n_tau, and τ for reflection)
- `o1`: First orbital dimension
- `o2`: Second orbital dimension (same as intermediate and output dimension)
"""
function ConvolutionWorkspace{T}(dlr_grid::DLRGrid, o1::Int, o2::Int) where {T<:Number}
    return ConvolutionWorkspace{T}(dlr_grid, o1, o2, o2)
end

"""
    compute_kernel_cache(dlr_grid::DLRGrid)

Pre-compute kernel_tau(0, omega[i], beta) for all DLR omega points,
and the convolution coefficient matrix values[k] / (ω[k] - ω[j]).

This cache eliminates ~1.1 GB of allocations in do_convolution_dlr2tau
and removes per-iteration division in the convolution inner loop.
"""
function compute_kernel_cache(dlr_grid::DLRGrid)
    r = length(dlr_grid.ω)
    ω = dlr_grid.ω

    # Compute kernel values at tau=0
    values = Vector{Float64}(undef, r)
    for i = 1:r
        values[i] = kernel_tau(0.0, ω[i], dlr_grid.β)
    end

    # Pre-compute convolution coefficients: values[k] / (ω[k] - ω[j]) for k ≠ j
    convolution_coeffs = zeros(Float64, r, r)
    @inbounds for k = 1:r
        for j = 1:r
            if k != j
                convolution_coeffs[k, j] = values[k] / (ω[k] - ω[j])
            end
            # diagonal (k == j) stays 0
        end
    end

    return KernelCache(values, convolution_coeffs)
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
