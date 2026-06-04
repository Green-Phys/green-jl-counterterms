"""
Matrix multiplication operations for Propagator representation.

These functions perform block-aware matrix multiplications along the τ-dimension:
- _multiply_op_prop!: Static operator × time-dependent tensor
- _multiply_prop_op!: Time-dependent tensor × static operator
- _trace_prop!: Trace operation over diagonal blocks

All operations respect quantum number conservation and only compute non-zero blocks.
"""

using QMBToolkitED: Basis, Operator
using LinearAlgebra

# Both Operator and Propagator carry a definite flux, so every block (tgt, src)
# satisfies QN(tgt) = QN(src) + flux. In a product result block (tgt, src) the
# intermediate sector is therefore fixed by the left factor's flux alone:
# QN(mid) = QN(tgt) - flux_left. This selects the single contributing mid via one
# qn_to_sector lookup instead of scanning every sector. Returns 0 if no such sector.
@inline function _intermediate_sector(basis::Basis, left_flux, tgt::Int)
    mid_qn = basis.sectors[tgt].qn + (-left_flux)
    return get(basis.qn_to_sector, mid_qn, 0)
end

"""
    _multiply_op_prop!(temp_buffer::Propagator{T}, A::Operator, B::Propagator{T}) where T<:Number

In-place operator-tensor multiplication: temp_buffer[:,:,τ] = A * B[:,:,τ] for all τ.

For each time slice τ, performs block-aware multiplication where only blocks allowed by
quantum number conservation are computed.

# Arguments

  - `temp_buffer::Propagator{T}`: Output buffer (will be overwritten)
  - `A::Operator`: Static operator (time-independent)
  - `B::Propagator{T}`: Time-dependent tensor

# Returns

  - `temp_buffer`: The input buffer with results written

# Algorithm

For each block (target, source) in the result:

  - Sum over intermediate sectors: result[tgt, src](%CF%84) = Σ_mid A[tgt, mid] * B[mid, src](%CF%84)
  - Uses native Operator multiplication at each time point
"""
function _multiply_op_prop!(
        temp_buffer::Propagator{T}, A::Operator,
        B::Propagator{T}) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Operator and array must use same basis"
    @boundscheck @assert temp_buffer.basis === B.basis "Buffer must use same basis as input"

    # With flux pre-filtering, propagators should never be empty
    @boundscheck @assert !isempty(B.blocks) "Unexpected empty propagator in multiply_op_prop!"

    # Get time dimension and basis
    # All blocks should have same n_tau, so get from any block (layout: dim1, dim2, n_tau)
    n_tau = size(first(values(B.blocks)), 3)
    basis = B.basis
    # Result flux is composition of input fluxes
    result_flux = A.flux + B.flux
    # Allow flux mismatch for empty buffers (they represent zero regardless of flux)
    @boundscheck @assert temp_buffer.flux == result_flux "Buffer flux must match result flux"

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid] * B[mid, src](τ)
    # The operator A is τ-independent, so all τ slices share the same left factor.
    # Reshaping B's block to (d_mid, d_src·n_tau) collapses the per-τ loop into a single GEMM.
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        d_tgt = size(result_block, 1)
        d_src = size(result_block, 2)
        result_mat = reshape(result_block, d_tgt, d_src * n_tau)

        # Single contributing intermediate sector, fixed by the left flux
        mid = _intermediate_sector(basis, A.flux, tgt)
        mid == 0 && continue
        A_key = (tgt, mid)
        B_key = (mid, src)
        (haskey(A.blocks, A_key) && haskey(B.blocks, B_key)) || continue
        A_block = A.blocks[A_key]
        B_block = B.blocks[B_key]

        # result[tgt, src] = A[tgt, mid] * B[mid, src] batched over all τ
        d_mid = size(B_block, 1)
        B_mat = reshape(B_block, d_mid, d_src * n_tau)
        mul!(result_mat, A_block, B_mat, one(T), one(T))
    end

    return temp_buffer
end

"""
    _multiply_prop_op!(temp_buffer::Propagator{T}, A::Propagator{T}, B::Operator) where T<:Number

In-place tensor-operator multiplication: temp_buffer[:,:,τ] = A[:,:,τ] * B for all τ.

For each time slice τ, performs block-aware multiplication where only blocks allowed by
quantum number conservation are computed.

# Arguments

  - `temp_buffer::Propagator{T}`: Output buffer (will be overwritten)
  - `A::Propagator{T}`: Time-dependent tensor
  - `B::Operator`: Static operator (time-independent)

# Returns

  - `temp_buffer`: The input buffer with results written

# Algorithm

For each block (target, source) in the result:

  - Sum over intermediate sectors: result[tgt, src](%CF%84) = Σ_mid A[tgt, mid](%CF%84) * B[mid, src]
"""
function _multiply_prop_op!(
        temp_buffer::Propagator{T}, A::Propagator{T},
        B::Operator) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Array and operator must use same basis"
    @boundscheck @assert temp_buffer.basis === A.basis "Buffer must use same basis as input"

    # With flux pre-filtering, propagators should never be empty
    @boundscheck @assert !isempty(A.blocks) "Unexpected empty propagator in multiply_prop_op!"

    # Get time dimension and basis (layout: dim1, dim2, n_tau)
    n_tau = size(first(values(A.blocks)), 3)
    basis = A.basis
    # Result flux is composition of input fluxes
    result_flux = A.flux + B.flux
    # Allow flux mismatch for empty buffers (they represent zero regardless of flux)
    @boundscheck @assert temp_buffer.flux == result_flux "Buffer flux must match result flux"

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src]
    # Use direct dict lookups instead of building temporary index structures
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Single contributing intermediate sector, fixed by the left flux
        mid = _intermediate_sector(basis, A.flux, tgt)
        mid == 0 && continue
        A_key = (tgt, mid)
        B_key = (mid, src)
        (haskey(A.blocks, A_key) && haskey(B.blocks, B_key)) || continue
        A_block = A.blocks[A_key]
        B_block = B.blocks[B_key]

        # For each time point: result[tgt, src](τ) = A[tgt, mid](τ) * B[mid, src]
        @inbounds for k in 1:n_tau
            mul!(view(result_block,:,:,k), view(A_block,:,:,k), B_block, 1.0, 1.0)
        end
    end

    return temp_buffer
end

"""
    _trace_prop!(trace_buffer::Vector{<:Real}, A::Propagator{<:Number})

In-place version: writes trace results to trace_buffer.

# Arguments

  - `trace_buffer::Vector{<:Real}`: Pre-allocated buffer for trace values
  - `A::Propagator{<:Number}`: Time-dependent block array

# Returns

  - `trace_buffer`: The input buffer with trace values written
"""
function _trace_prop!(trace_buffer::Vector{<:Real}, A::Propagator{<:Number})
    # With flux pre-filtering, propagators should never be empty
    @boundscheck @assert !isempty(A.blocks) "Unexpected empty propagator in trace_prop!"

    # Get time dimension from any block (layout: dim1, dim2, n_tau)
    n_tau = size(first(values(A.blocks)), 3)
    @boundscheck @assert length(trace_buffer) == n_tau "Trace buffer must have length n_tau"

    # Zero out buffer
    fill!(trace_buffer, 0.0)

    # Sum traces of diagonal blocks only
    for ((target, source), block) in A.blocks
        if target == source  # Diagonal block
            dim = size(block, 1)
            @boundscheck @assert size(block, 2) == dim "Diagonal blocks must be square"

            # Manual diagonal sum is faster than tr(@view ...) for small blocks
            # Loop order: k outer, i inner with @simd for vectorization
            @inbounds for k in 1:n_tau
                @simd for i in 1:dim
                    trace_buffer[k] += real(block[i, i, k])
                end
            end
        end
    end

    return trace_buffer
end

"""
    _multiply_prop_prop!(temp_buffer::Propagator{T}, A::Propagator{T}, B::Propagator{T}) where T<:Number

In-place tensor-tensor multiplication: temp_buffer[:,:,τ] = A[:,:,τ] * B[:,:,τ].

For each time slice τ, performs block-aware multiplication where only blocks allowed by
quantum number conservation are computed.

# Arguments

  - `temp_buffer::Propagator{T}`: Output buffer (will be overwritten)
  - `A::Propagator{T}`: First time-dependent tensor
  - `B::Propagator{T}`: Second time-dependent tensor

# Returns

  - `temp_buffer`: The input buffer with results written

# Algorithm

For each block (target, source) in the result:

  - Sum over intermediate sectors: result[tgt, src](%CF%84) = Σ_mid A[tgt, mid](%CF%84) * B[mid, src](%CF%84)
"""
function _multiply_prop_prop!(
        temp_buffer::Propagator{T}, A::Propagator{T},
        B::Propagator{T}) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Both arrays must use same basis"
    @boundscheck @assert temp_buffer.basis === A.basis "Buffer must use same basis as inputs"

    # With flux pre-filtering, propagators should never be empty
    @boundscheck @assert !isempty(A.blocks) "Unexpected empty propagator A in multiply_prop_prop!"
    @boundscheck @assert !isempty(B.blocks) "Unexpected empty propagator B in multiply_prop_prop!"

    # Get time dimension and basis (layout: dim1, dim2, n_tau)
    n_tau = size(first(values(A.blocks)), 3)
    basis = A.basis
    # Result flux is composition of input fluxes
    result_flux = A.flux + B.flux
    @boundscheck @assert temp_buffer.flux == result_flux "Buffer flux must match result flux"

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src](τ)
    # Use direct dict lookups instead of building temporary index structures
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Single contributing intermediate sector, fixed by the left flux
        mid = _intermediate_sector(basis, A.flux, tgt)
        mid == 0 && continue
        A_key = (tgt, mid)
        B_key = (mid, src)
        (haskey(A.blocks, A_key) && haskey(B.blocks, B_key)) || continue
        A_block = A.blocks[A_key]
        B_block = B.blocks[B_key]

        # For each time point: result[tgt, src](τ) = A[tgt, mid](τ) * B[mid, src](τ)
        @inbounds for k in 1:n_tau
            mul!(view(result_block,:,:,k), view(A_block,:,:,k),
                view(B_block,:,:,k), 1.0, 1.0)
        end
    end

    return temp_buffer
end

# ============================================================================
# Fused K-factor × Matrix Multiplication Operations
# ============================================================================

"""
    _multiply_op_prop_with_K!(result, A, B, K)

Fused operator-propagator multiplication with K-factor scaling:
result[:,:,k] = A * B[:,:,k] * K[k]

Combines matrix multiplication with K-factor scaling in one function,
saving one memory pass compared to separate multiply + hybridization calls.

# Arguments

  - `result::Propagator{T}`: Output buffer (will be overwritten)
  - `A::Operator`: Static operator (time-independent)
  - `B::Propagator{T}`: Time-dependent tensor (input)
  - `K::AbstractVector{Float64}`: K-factor vector to apply after matmul
"""
function _multiply_op_prop_with_K!(
        result::Propagator{T}, A::Operator, B::Propagator{T},
        K::AbstractVector{Float64}) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Operator and array must use same basis"
    @boundscheck @assert result.basis === B.basis "Buffer must use same basis as input"
    @boundscheck @assert !isempty(B.blocks) "Unexpected empty propagator"

    n_tau = size(first(values(B.blocks)), 3)
    basis = B.basis
    result_flux = A.flux + B.flux
    @boundscheck @assert result.flux == result_flux "Buffer flux must match result flux"

    for (result_key, result_block) in result.blocks
        (tgt, src) = result_key
        fill!(result_block, zero(T))

        d_tgt = size(result_block, 1)
        d_src = size(result_block, 2)
        result_mat = reshape(result_block, d_tgt, d_src * n_tau)

        # Single contributing intermediate sector, fixed by the left flux
        mid = _intermediate_sector(basis, A.flux, tgt)
        if mid != 0
            A_key = (tgt, mid)
            B_key = (mid, src)
            if haskey(A.blocks, A_key) && haskey(B.blocks, B_key)
                A_block = A.blocks[A_key]
                B_block = B.blocks[B_key]

                # result = A * B batched over all τ in one GEMM
                d_mid = size(B_block, 1)
                B_mat = reshape(B_block, d_mid, d_src * n_tau)
                mul!(result_mat, A_block, B_mat, one(T), one(T))
            end
        end

        # K-factor scaling: result *= K
        @inbounds for k in 1:n_tau
            @views result_block[:, :, k] .*= K[k]
        end
    end

    return result
end

"""
    _multiply_prop_op_with_K!(result, A, B, K)

Fused propagator-operator multiplication with K-factor scaling:
result[:,:,k] = A[:,:,k] * B * K[k]

Combines matrix multiplication with K-factor scaling in one function.

# Arguments

  - `result::Propagator{T}`: Output buffer (will be overwritten)
  - `A::Propagator{T}`: Time-dependent tensor (input)
  - `B::Operator`: Static operator (time-independent)
  - `K::AbstractVector{Float64}`: K-factor vector to apply after matmul
"""
function _multiply_prop_op_with_K!(
        result::Propagator{T}, A::Propagator{T}, B::Operator,
        K::AbstractVector{Float64}) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Array and operator must use same basis"
    @boundscheck @assert result.basis === A.basis "Buffer must use same basis as input"
    @boundscheck @assert !isempty(A.blocks) "Unexpected empty propagator"

    n_tau = size(first(values(A.blocks)), 3)
    basis = A.basis
    result_flux = A.flux + B.flux
    @boundscheck @assert result.flux == result_flux "Buffer flux must match result flux"

    for (result_key, result_block) in result.blocks
        (tgt, src) = result_key
        fill!(result_block, zero(T))

        # Single contributing intermediate sector, fixed by the left flux
        mid = _intermediate_sector(basis, A.flux, tgt)
        if mid != 0
            A_key = (tgt, mid)
            B_key = (mid, src)
            if haskey(A.blocks, A_key) && haskey(B.blocks, B_key)
                A_block = A.blocks[A_key]
                B_block = B.blocks[B_key]

                @inbounds for k in 1:n_tau
                    mul!(view(result_block,:,:,k), view(A_block,:,:,k), B_block, 1.0,
                        1.0)
                end
            end
        end

        # K-factor scaling: result *= K
        @inbounds for k in 1:n_tau
            @views result_block[:, :, k] .*= K[k]
        end
    end

    return result
end

# ============================================================================
# Interface Implementations
# ============================================================================

# Operator-propagator multiplication: result = op * prop
function multiply_op_prop!(result::Propagator{T}, op::Operator,
        prop::Propagator{T}) where {T}
    _multiply_op_prop!(result, op, prop)
end

# Propagator-operator multiplication: result = prop * op
function multiply_prop_op!(result::Propagator{T}, prop::Propagator{T},
        op::Operator) where {T}
    _multiply_prop_op!(result, prop, op)
end

# Propagator-propagator multiplication: result = A * B
function multiply_prop_prop!(result::Propagator{T}, A::Propagator{T},
        B::Propagator{T}) where {T}
    _multiply_prop_prop!(result, A, B)
end

# Trace over spatial indices
function trace_prop!(result_buffer::Vector{<:Real}, prop::Propagator)
    _trace_prop!(result_buffer, prop)
end

# Fused operator-propagator multiplication with K-factor
function multiply_op_prop_with_K!(result::Propagator{T}, op::Operator,
        prop::Propagator{T}, K::AbstractVector{Float64}) where {T}
    _multiply_op_prop_with_K!(result, op, prop, K)
end

# Fused propagator-operator multiplication with K-factor
function multiply_prop_op_with_K!(result::Propagator{T}, prop::Propagator{T},
        op::Operator, K::AbstractVector{Float64}) where {T}
    _multiply_prop_op_with_K!(result, prop, op, K)
end
