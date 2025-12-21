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
function _multiplication_23_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDenseOperator, B::BlockDensePropagator{T}) where T<:Number
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
    B_by_mid = Dict{Int, Vector{Tuple{Int, Array{T,3}}}}()
    for ((mid, src), B_block) in B.blocks
        if !haskey(B_by_mid, mid)
            B_by_mid[mid] = Tuple{Int, Array{T,3}}[]
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
                        mul!(view(result_block, k, :, :), A_block, view(B_block, k, :, :), 1.0, 1.0)
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
function _multiplication_32_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T}, B::BlockDenseOperator) where T<:Number
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
    A_by_mid = Dict{Int, Vector{Tuple{Int, Array{T,3}}}}()
    for ((tgt, mid), A_block) in A.blocks
        if !haskey(A_by_mid, mid)
            A_by_mid[mid] = Tuple{Int, Array{T,3}}[]
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
                        mul!(view(result_block, k, :, :), view(A_block, k, :, :), B_block, 1.0, 1.0)
                    end
                end
            end
        end
    end

    return temp_buffer
end

"""
    _trace_3_blockdense(A::BlockDensePropagator{T}) where T<:Number

Compute trace of each τ-slice: result[τ] = tr(A[τ,:,:]).

For BlockDense arrays, the trace is the sum of traces of diagonal blocks:
    tr(A[τ]) = Σ_sector tr(A[sector, sector](τ))

# Arguments
- `A::BlockDensePropagator{T}`: Time-dependent block-dense array

# Returns
- `Vector{TReal}`: Trace at each time point (always real even for complex matrices)

# Note
Only diagonal blocks contribute to the trace (quantum number conservation).
"""
function _trace_3_blockdense(A::BlockDensePropagator{T}) where T<:Number
    # Get time dimension from any block
    n_tau = size(first(values(A.blocks)), 1)

    TReal = real(T)  # Trace is always real even for complex matrices
    trace_vals = zeros(TReal, n_tau)

    # Sum traces of diagonal blocks only
    for ((target, source), block) in A.blocks
        if target == source  # Diagonal block
            dim = size(block, 2)
            @assert size(block, 3) == dim "Diagonal blocks must be square"

            @inbounds for k in 1:n_tau
                trace_vals[k] += tr(block[k, :, :])
            end
        end
    end

    return trace_vals
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
                trace_buffer[k] += tr(block[k, :, :])
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
function _multiplication_33_blockdense!(temp_buffer::BlockDensePropagator{T}, A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T<:Number
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
    B_by_mid = Dict{Int, Vector{Tuple{Int, Array{T,3}}}}()
    for ((mid, src), B_block) in B.blocks
        if !haskey(B_by_mid, mid)
            B_by_mid[mid] = Tuple{Int, Array{T,3}}[]
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
                        mul!(view(result_block, k, :, :), view(A_block, k, :, :), view(B_block, k, :, :), 1.0, 1.0)
                    end
                end
            end
        end
    end

    return temp_buffer
end
