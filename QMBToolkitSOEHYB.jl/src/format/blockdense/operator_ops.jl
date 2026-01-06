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

In-place operator-tensor multiplication: temp_buffer[τ,:,:] = A * B[τ,:,:] for all τ.

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
- Sum over intermediate sectors: result[tgt, src](τ) = Σ_mid A[tgt, mid] * B[mid, src](τ)
- Uses native BlockDenseOperator multiplication at each time point
"""
function _multiplication_23_blockdense!(
        temp_buffer::BlockDensePropagator{T}, A::BlockDenseOperator,
        B::BlockDensePropagator{T},) where {T <: Number}
    @assert A.basis === B.basis "Operator and array must use same basis"
    @assert temp_buffer.basis === B.basis "Buffer must use same basis as input"

    # Handle empty B case (propagator with no blocks)
    if isempty(B.blocks)
        # Result is also empty - fill temp_buffer blocks with zeros
        for (key, block) in temp_buffer.blocks
            fill!(block, zero(T))
        end
        return
    end

    # Get time dimension
    # All blocks should have same n_tau, so get from any block
    n_tau = size(first(values(B.blocks)), 1)

    # Result flux is composition of input fluxes
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))
    @assert temp_buffer.flux.delta_qn == result_flux.delta_qn "Buffer flux must match result flux"

    # Index B blocks by their target (mid) sector for efficient lookup
    B_by_mid = Dict{Int, Vector{Tuple{Int, Array{T, 3}}}}()
    for ((mid, src), B_block) in B.blocks
        if !haskey(B_by_mid, mid)
            B_by_mid[mid] = Tuple{Int, Array{T, 3}}[]
        end
        push!(B_by_mid[mid], (src, B_block))
    end

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid] * B[mid, src](τ)
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Accumulate contributions from all intermediate sectors
        for ((A_tgt, mid), A_block) in A.blocks
            if A_tgt != tgt
                continue
            end

            if haskey(B_by_mid, mid)
                for (B_src, B_block) in B_by_mid[mid]
                    if B_src != src
                        continue
                    end

                    # For each time point: result[tgt, src](τ) += A[tgt, mid] * B[mid, src](τ)
                    @inbounds for k in 1:n_tau
                        mul!(view(result_block,k,:,:), A_block, view(B_block,k,:,:), 1.0, 1.0)
                    end
                end
            end
        end
    end

    return temp_buffer
end

"""
    _multiplication_32_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T}, B::BlockDenseOperator) where T<:Number

In-place tensor-operator multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B for all τ.

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
- Sum over intermediate sectors: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src]
"""
function _multiplication_32_blockdense!(
        temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T},
        B::BlockDenseOperator,) where {T <: Number}
    @assert A.basis === B.basis "Array and operator must use same basis"
    @assert temp_buffer.basis === A.basis "Buffer must use same basis as input"

    # Handle empty A case
    if isempty(A.blocks)
        for (key, block) in temp_buffer.blocks
            fill!(block, zero(T))
        end
        return
    end

    # Get time dimension
    n_tau = size(first(values(A.blocks)), 1)

    # Result flux is composition of input fluxes
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))
    @assert temp_buffer.flux.delta_qn == result_flux.delta_qn "Buffer flux must match result flux"

    # Index A blocks by their source (mid) sector for efficient lookup
    A_by_mid = Dict{Int, Vector{Tuple{Int, Array{T, 3}}}}()
    for ((tgt, mid), A_block) in A.blocks
        if !haskey(A_by_mid, mid)
            A_by_mid[mid] = Tuple{Int, Array{T, 3}}[]
        end
        push!(A_by_mid[mid], (tgt, A_block))
    end

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src]
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Accumulate contributions from all intermediate sectors
        for ((mid, B_src), B_block) in B.blocks
            if B_src != src
                continue
            end

            if haskey(A_by_mid, mid)
                for (A_tgt, A_block) in A_by_mid[mid]
                    if A_tgt != tgt
                        continue
                    end

                    # For each time point: result[tgt, src](τ) += A[tgt, mid](τ) * B[mid, src]
                    @inbounds for k in 1:n_tau
                        mul!(view(result_block,k,:,:), view(A_block,k,:,:), B_block, 1.0, 1.0)
                    end
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

    # Get time dimension from any block
    n_tau = size(first(values(A.blocks)), 1)
    @assert length(trace_buffer) == n_tau "Trace buffer must have length n_tau"

    # Zero out buffer
    fill!(trace_buffer, 0.0)

    # Sum traces of diagonal blocks only
    for ((target, source), block) in A.blocks
        if target == source  # Diagonal block
            dim = size(block, 2)
            @assert size(block, 3) == dim "Diagonal blocks must be square"

            @inbounds for k in 1:n_tau
                trace_buffer[k] += tr(@view block[k, :, :])
            end
        end
    end

    return trace_buffer
end

"""
    _multiplication_33_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T<:Number

In-place tensor-tensor multiplication for BlockDense: temp_buffer[τ,:,:] = A[τ,:,:] * B[τ,:,:].

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
- Sum over intermediate sectors: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src](τ)
"""
function _multiplication_33_blockdense!(
        temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T},
        B::BlockDensePropagator{T},) where {T <: Number}
    @assert A.basis === B.basis "Both arrays must use same basis"
    @assert temp_buffer.basis === A.basis "Buffer must use same basis as inputs"

    # Handle empty A or B case
    if isempty(A.blocks) || isempty(B.blocks)
        for (key, block) in temp_buffer.blocks
            fill!(block, zero(T))
        end
        return temp_buffer
    end

    # Get time dimension
    n_tau = size(first(values(A.blocks)), 1)

    # Result flux is composition of input fluxes
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))
    @assert temp_buffer.flux.delta_qn == result_flux.delta_qn "Buffer flux must match result flux"

    # Index B blocks by their target (mid) sector for efficient lookup
    B_by_mid = Dict{Int, Vector{Tuple{Int, Array{T, 3}}}}()
    for ((mid, src), B_block) in B.blocks
        if !haskey(B_by_mid, mid)
            B_by_mid[mid] = Tuple{Int, Array{T, 3}}[]
        end
        push!(B_by_mid[mid], (src, B_block))
    end

    # For each result block, compute: result[tgt, src](τ) = Σ_mid A[tgt, mid](τ) * B[mid, src](τ)
    for (result_key, result_block) in temp_buffer.blocks
        (tgt, src) = result_key

        # Zero out the buffer block
        fill!(result_block, zero(T))

        # Accumulate contributions from all intermediate sectors
        for ((A_tgt, mid), A_block) in A.blocks
            if A_tgt != tgt
                continue
            end

            if haskey(B_by_mid, mid)
                for (B_src, B_block) in B_by_mid[mid]
                    if B_src != src
                        continue
                    end

                    # For each time point: result[tgt, src](τ) += A[tgt, mid](τ) * B[mid, src](τ)
                    @inbounds for k in 1:n_tau
                        mul!(view(result_block,k,:,:), view(A_block,k,:,:),
                            view(B_block,k,:,:), 1.0, 1.0,)
                    end
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
        prop::BlockDensePropagator{T},) where {T}
    _multiplication_23_blockdense!(result, op, prop)
end

# Propagator-operator multiplication: result = prop * op
function multiply_prop_op!(result::BlockDensePropagator{T}, prop::BlockDensePropagator{T},
        op::BlockDenseOperator,) where {T}
    _multiplication_32_blockdense!(result, prop, op)
end

# Propagator-propagator multiplication: result = A * B
function multiply_prop_prop!(result::BlockDensePropagator{T}, A::BlockDensePropagator{T},
        B::BlockDensePropagator{T},) where {T}
    _multiplication_33_blockdense!(result, A, B)
end

# Trace over spatial indices
function trace_prop!(result_buffer::Vector{<:Real}, prop::BlockDensePropagator)
    _trace_3_blockdense!(result_buffer, prop)
end

#=============================================================================
# Product Buffer Allocation Helpers
=============================================================================#

"""
    allocate_product_buffer(op::BlockDenseOperator, prop::BlockDensePropagator{T}, n_tau::Int) where T

Allocate a buffer for the result of op * prop (operator × propagator multiplication).
Determines which blocks exist based on quantum number conservation.
"""
function allocate_product_buffer(op::BlockDenseOperator, prop::BlockDensePropagator{T}, n_tau::Int) where {T}
    basis = prop.basis
    op_flux = OperatorFlux(combine_qn(op.flux.delta_qn, prop.flux.delta_qn))

    result_blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    for ((tgt, mid), op_block) in op.blocks
        for ((mid2, src), prop_block) in prop.blocks
            if mid == mid2
                key = (tgt, src)
                if !haskey(result_blocks, key)
                    dim_tgt = basis.sectors[tgt].dim
                    dim_src = basis.sectors[src].dim
                    result_blocks[key] = zeros(T, n_tau, dim_tgt, dim_src)
                end
            end
        end
    end
    return BlockDensePropagator{T}(basis, op_flux, result_blocks)
end

"""
    allocate_product_buffer(prop::BlockDensePropagator{T}, op::BlockDenseOperator, n_tau::Int) where T

Allocate a buffer for the result of prop * op (propagator × operator multiplication).
"""
function allocate_product_buffer(prop::BlockDensePropagator{T}, op::BlockDenseOperator, n_tau::Int) where {T}
    basis = prop.basis
    op_flux = OperatorFlux(combine_qn(prop.flux.delta_qn, op.flux.delta_qn))

    result_blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    for ((tgt, mid), prop_block) in prop.blocks
        for ((mid2, src), op_block) in op.blocks
            if mid == mid2
                key = (tgt, src)
                if !haskey(result_blocks, key)
                    dim_tgt = basis.sectors[tgt].dim
                    dim_src = basis.sectors[src].dim
                    result_blocks[key] = zeros(T, n_tau, dim_tgt, dim_src)
                end
            end
        end
    end
    return BlockDensePropagator{T}(basis, op_flux, result_blocks)
end

"""
    allocate_product_buffer(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}, n_tau::Int) where T

Allocate a buffer for the result of A * B (propagator × propagator multiplication).
"""
function allocate_product_buffer(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}, n_tau::Int) where {T}
    basis = A.basis
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))

    result_blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    for ((tgt, mid), A_block) in A.blocks
        for ((mid2, src), B_block) in B.blocks
            if mid == mid2
                key = (tgt, src)
                if !haskey(result_blocks, key)
                    dim_tgt = basis.sectors[tgt].dim
                    dim_src = basis.sectors[src].dim
                    result_blocks[key] = zeros(T, n_tau, dim_tgt, dim_src)
                end
            end
        end
    end
    return BlockDensePropagator{T}(basis, result_flux, result_blocks)
end
