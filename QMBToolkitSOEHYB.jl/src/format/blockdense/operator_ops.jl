"""
Matrix multiplication operations for BlockDense representation.

These functions perform block-aware matrix multiplications along the τ-dimension:
- _multiplication_23_blockdense!: Static operator × time-dependent tensor
- _multiplication_32_blockdense!: Time-dependent tensor × static operator
- _trace_3_blockdense: Trace operation over diagonal blocks

All operations respect quantum number conservation and only compute non-zero blocks.
"""

using QMBToolkitED.BlockDense
using LinearAlgebra

# BlockDensePropagator is defined in solver/Propagator.jl

"""
    _multiplication_23_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDenseOperator, B::BlockDensePropagator{T}) where T<:Number

In-place operator-tensor multiplication: temp_buffer[:,:,τ] = A * B[:,:,τ] for all τ.

For each time slice τ, performs block-aware multiplication where only blocks allowed by
quantum number conservation are computed.

# Arguments

  - `temp_buffer::BlockDensePropagator{T}`: Output buffer (will be overwritten)
  - `A::BlockDenseOperator`: Static operator (time-independent)
  - `B::BlockDensePropagator{T}`: Time-dependent tensor

# Returns

  - `temp_buffer`: The input buffer with results written

# Algorithm

For each block (target, source) in the result:

  - Sum over intermediate sectors: result[tgt, src](%CF%84) = Σ_mid A[tgt, mid] * B[mid, src](%CF%84)
  - Uses native BlockDenseOperator multiplication at each time point
"""
function _multiplication_23_blockdense!(
        temp_buffer::BlockDensePropagator{T}, A::BlockDenseOperator,
        B::BlockDensePropagator{T}) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Operator and array must use same basis"
    @boundscheck @assert temp_buffer.basis === B.basis "Buffer must use same basis as input"

    # Handle empty B case (propagator with no blocks)
    if isempty(B.blocks)
        # Result is also empty - fill temp_buffer blocks with zeros
        for (key, block) in temp_buffer.blocks
            fill!(block, zero(T))
        end
        return
    end

    # Get time dimension and basis
    # All blocks should have same n_tau, so get from any block (layout: dim1, dim2, n_tau)
    n_tau = size(first(values(B.blocks)), 3)
    basis = B.basis
    n_sectors = length(basis.sectors)

    # Result flux is composition of input fluxes
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))
    @boundscheck @assert temp_buffer.flux.delta_qn == result_flux.delta_qn "Buffer flux must match result flux"

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid] * B[mid, src](τ)
    # Use direct dict lookups instead of building temporary index structures
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Sum over all intermediate sectors with direct O(1) lookups
        @inbounds for mid in 1:n_sectors
            A_key = (tgt, mid)
            B_key = (mid, src)

            # Direct dict lookups - no allocation, O(1) each
            if haskey(A.blocks, A_key) && haskey(B.blocks, B_key)
                A_block = A.blocks[A_key]
                B_block = B.blocks[B_key]

                # For each time point: result[tgt, src](τ) += A[tgt, mid] * B[mid, src](τ)
                @inbounds for k in 1:n_tau
                    mul!(view(result_block,:,:,k), A_block, view(B_block,:,:,k), 1.0, 1.0)
                end
            end
        end
    end

    return temp_buffer
end

"""
    _multiplication_32_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T}, B::BlockDenseOperator) where T<:Number

In-place tensor-operator multiplication: temp_buffer[:,:,τ] = A[:,:,τ] * B for all τ.

For each time slice τ, performs block-aware multiplication where only blocks allowed by
quantum number conservation are computed.

# Arguments

  - `temp_buffer::BlockDensePropagator{T}`: Output buffer (will be overwritten)
  - `A::BlockDensePropagator{T}`: Time-dependent tensor
  - `B::BlockDenseOperator`: Static operator (time-independent)

# Returns

  - `temp_buffer`: The input buffer with results written

# Algorithm

For each block (target, source) in the result:

  - Sum over intermediate sectors: result[tgt, src](%CF%84) = Σ_mid A[tgt, mid](%CF%84) * B[mid, src]
"""
function _multiplication_32_blockdense!(
        temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T},
        B::BlockDenseOperator) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Array and operator must use same basis"
    @boundscheck @assert temp_buffer.basis === A.basis "Buffer must use same basis as input"

    # Handle empty A case
    if isempty(A.blocks)
        for (key, block) in temp_buffer.blocks
            fill!(block, zero(T))
        end
        return
    end

    # Get time dimension and basis (layout: dim1, dim2, n_tau)
    n_tau = size(first(values(A.blocks)), 3)
    basis = A.basis
    n_sectors = length(basis.sectors)

    # Result flux is composition of input fluxes
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))
    @boundscheck @assert temp_buffer.flux.delta_qn == result_flux.delta_qn "Buffer flux must match result flux"

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src]
    # Use direct dict lookups instead of building temporary index structures
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Sum over all intermediate sectors with direct O(1) lookups
        @inbounds for mid in 1:n_sectors
            A_key = (tgt, mid)
            B_key = (mid, src)

            # Direct dict lookups - no allocation, O(1) each
            if haskey(A.blocks, A_key) && haskey(B.blocks, B_key)
                A_block = A.blocks[A_key]
                B_block = B.blocks[B_key]

                # For each time point: result[tgt, src](τ) += A[tgt, mid](τ) * B[mid, src]
                @inbounds for k in 1:n_tau
                    mul!(view(result_block,:,:,k), view(A_block,:,:,k), B_block, 1.0, 1.0)
                end
            end
        end
    end

    return temp_buffer
end

"""
    _trace_3_blockdense!(trace_buffer::Vector{<:Real}, A::BlockDensePropagator{<:Number})

In-place version: writes trace results to trace_buffer.

# Arguments

  - `trace_buffer::Vector{<:Real}`: Pre-allocated buffer for trace values
  - `A::BlockDensePropagator{<:Number}`: Time-dependent block-dense array

# Returns

  - `trace_buffer`: The input buffer with trace values written
"""
function _trace_3_blockdense!(trace_buffer::Vector{<:Real}, A::BlockDensePropagator{<:Number})
    # Handle empty case
    if isempty(A.blocks)
        fill!(trace_buffer, 0.0)
        return trace_buffer
    end

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
    _multiplication_33_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T<:Number

In-place tensor-tensor multiplication for BlockDense: temp_buffer[:,:,τ] = A[:,:,τ] * B[:,:,τ].

For each time slice τ, performs block-aware multiplication where only blocks allowed by
quantum number conservation are computed.

# Arguments

  - `temp_buffer::BlockDensePropagator{T}`: Output buffer (will be overwritten)
  - `A::BlockDensePropagator{T}`: First time-dependent tensor
  - `B::BlockDensePropagator{T}`: Second time-dependent tensor

# Returns

  - `temp_buffer`: The input buffer with results written

# Algorithm

For each block (target, source) in the result:

  - Sum over intermediate sectors: result[tgt, src](%CF%84) = Σ_mid A[tgt, mid](%CF%84) * B[mid, src](%CF%84)
"""
function _multiplication_33_blockdense!(
        temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T},
        B::BlockDensePropagator{T}) where {T <: Number}
    @boundscheck @assert A.basis === B.basis "Both arrays must use same basis"
    @boundscheck @assert temp_buffer.basis === A.basis "Buffer must use same basis as inputs"

    # Handle empty A or B case
    if isempty(A.blocks) || isempty(B.blocks)
        for (key, block) in temp_buffer.blocks
            fill!(block, zero(T))
        end
        return temp_buffer
    end

    # Get time dimension and basis (layout: dim1, dim2, n_tau)
    n_tau = size(first(values(A.blocks)), 3)
    basis = A.basis
    n_sectors = length(basis.sectors)

    # Result flux is composition of input fluxes
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))
    @boundscheck @assert temp_buffer.flux.delta_qn == result_flux.delta_qn "Buffer flux must match result flux"

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src](τ)
    # Use direct dict lookups instead of building temporary index structures
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Sum over all intermediate sectors with direct O(1) lookups
        @inbounds for mid in 1:n_sectors
            A_key = (tgt, mid)
            B_key = (mid, src)

            # Direct dict lookups - no allocation, O(1) each
            if haskey(A.blocks, A_key) && haskey(B.blocks, B_key)
                A_block = A.blocks[A_key]
                B_block = B.blocks[B_key]

                # For each time point: result[tgt, src](τ) += A[tgt, mid](τ) * B[mid, src](τ)
                @inbounds for k in 1:n_tau
                    mul!(view(result_block,:,:,k), view(A_block,:,:,k),
                        view(B_block,:,:,k), 1.0, 1.0)
                end
            end
        end
    end

    return temp_buffer
end

# ============================================================================
# Interface Implementations
# ============================================================================

# Operator-propagator multiplication: result = op * prop
function multiply_op_prop!(result::BlockDensePropagator{T}, op::BlockDenseOperator,
        prop::BlockDensePropagator{T}) where {T}
    _multiplication_23_blockdense!(result, op, prop)
end

# Propagator-operator multiplication: result = prop * op
function multiply_prop_op!(result::BlockDensePropagator{T}, prop::BlockDensePropagator{T},
        op::BlockDenseOperator) where {T}
    _multiplication_32_blockdense!(result, prop, op)
end

# Propagator-propagator multiplication: result = A * B
function multiply_prop_prop!(result::BlockDensePropagator{T}, A::BlockDensePropagator{T},
        B::BlockDensePropagator{T}) where {T}
    _multiplication_33_blockdense!(result, A, B)
end

# Trace over spatial indices
function trace_prop!(result_buffer::Vector{<:Real}, prop::BlockDensePropagator)
    _trace_3_blockdense!(result_buffer, prop)
end
