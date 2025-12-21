"""
Matrix multiplication operations for dense representation.

These functions perform various matrix multiplications along the τ-dimension:
- _multiplication_23: Operator-propagator multiplication (DenseOperator × DensePropagator)
- _multiplication_32: Propagator-operator multiplication (DensePropagator × DenseOperator)
- _multiplication_13: Vector-propagator multiplication (1D vector × DensePropagator, element-wise)
- _multiplication_33: Propagator-propagator multiplication (DensePropagator × DensePropagator)
- _trace_3: Trace operation over τ-slices of DensePropagator

Each operation has both in-place (!) and allocating versions.
Works with both wrapper types (DenseOperator, DensePropagator) and raw arrays.
"""

using LinearAlgebra
using QMBToolkitED.Dense: DenseOperator

# ============================================================================
# Matrix-Tensor Multiplications: (τ, i, j) shaped arrays
# ============================================================================

"""
    _multiplication_23!(temp_buffer::Array{T,3}, A::Array{T,2}, B::Array{T,3}) where T<:Number

In-place matrix-tensor multiplication: temp_buffer[τ,:,:] = A * B[τ,:,:] for all τ.
Writes result to temp_buffer. Caller should copy back to input if needed.
"""
function _multiplication_23!(temp_buffer::Array{T,3}, A::Array{T,2}, B::Array{T,3}) where T<:Number
    n_tau = size(B, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer, k, :, :), A, view(B, k, :, :))
    end
    return temp_buffer
end

"""
    _multiplication_23(A::Array{T,2}, B::Array{T,3}) where T<:Number

Allocating version: creates new array for result.
"""
function _multiplication_23(A::Array{T,2}, B::Array{T,3}) where T<:Number
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(T, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A * B[k, :, :]
    end
    return C
end

"""
    _multiplication_32!(temp_buffer::Array{T,3}, A::Array{T,3}, B::Array{T,2}) where T<:Number

In-place tensor-matrix multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B for all τ.
"""
function _multiplication_32!(temp_buffer::Array{T,3}, A::Array{T,3}, B::Array{T,2}) where T<:Number
    n_tau = size(A, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer, k, :, :), view(A, k, :, :), B)
    end
    return temp_buffer
end

"""
    _multiplication_32(A::Array{T,3}, B::Array{T,2}) where T<:Number

Allocating version.
"""
function _multiplication_32(A::Array{T,3}, B::Array{T,2}) where T<:Number
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    C = zeros(T, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k, :, :] * B
    end
    return C
end

"""
    _multiplication_13!(temp_buffer::Array{T,3}, A::AbstractVector, B::Array{T,3}) where T<:Number

In-place vector-tensor element-wise multiplication: temp_buffer[τ,:,:] = A[τ] * B[τ,:,:].
"""
function _multiplication_13!(temp_buffer::Array{T,3}, A::AbstractVector, B::Array{T,3}) where T<:Number
    n_tau = size(B, 1)
    @inbounds for k in 1:n_tau
        @views temp_buffer[k, :, :] .= A[k] .* B[k, :, :]
    end
    return temp_buffer
end

"""
    _multiplication_13(A::Array{<:Number,1}, B::Array{T,3}) where T<:Number

Allocating version.
"""
function _multiplication_13(A::Array{<:Number,1}, B::Array{T,3}) where T<:Number
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(T, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k] .* B[k, :, :]
    end
    return C
end

"""
    _multiplication_33!(temp_buffer::Array{T,3}, A::Array{T,3}, B::Array{T,3}) where T<:Number

In-place tensor-tensor multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B[τ,:,:].
"""
function _multiplication_33!(temp_buffer::Array{T,3}, A::Array{T,3}, B::Array{T,3}) where T<:Number
    n_tau = size(A, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer, k, :, :), view(A, k, :, :), view(B, k, :, :))
    end
    return temp_buffer
end

"""
    _multiplication_33(A::Array{T,3}, B::Array{T,3}) where T<:Number

Allocating version.
"""
function _multiplication_33(A::Array{T,3}, B::Array{T,3}) where T<:Number
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(T, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k, :, :] * B[k, :, :]
    end
    return C
end

# ============================================================================
# Trace Operations
# ============================================================================

"""
    _trace_3(A::Array{T,3}) where T<:Number

Compute trace of each τ-slice: result[τ] = tr(A[τ,:,:]).
"""
function _trace_3(A::Array{T,3}) where T<:Number
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    o1 == o2 || error("Matrix to be traced must be square: got dimensions $o1 x $o2")
    TReal = real(T)  # Trace is always real even for complex matrices
    trace_vals = zeros(TReal, n_tau)
    for k in 1:n_tau
        trace_vals[k] = tr(A[k, :, :])
    end
    return trace_vals
end

"""
    _trace_3!(trace_buffer::Vector{<:Real}, A::Array{<:Number,3})

In-place version: writes trace results to trace_buffer.
"""
function _trace_3!(trace_buffer::Vector{<:Real}, A::Array{<:Number,3})
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    o1 == o2 || error("Matrix to be traced must be square: got dimensions $o1 x $o2")
    @inbounds for k in 1:n_tau
        trace_buffer[k] = tr(A[k, :, :])
    end
    return trace_buffer
end

# ============================================================================
# Wrapper Methods for DenseOperator and DensePropagator
# ============================================================================

"""
    _multiplication_23!(temp_buffer::DensePropagator{T}, A::DenseOperator, B::DensePropagator{T}) where T<:Number

In-place operator-propagator multiplication: temp_buffer[τ,:,:] = A * B[τ,:,:] for all τ.
"""
function _multiplication_23!(temp_buffer::DensePropagator{T}, A::DenseOperator, B::DensePropagator{T}) where T<:Number
    _multiplication_23!(temp_buffer.data, A.data, B.data)
    return temp_buffer
end

"""
    _multiplication_32!(temp_buffer::DensePropagator{T}, A::DensePropagator{T}, B::DenseOperator) where T<:Number

In-place propagator-operator multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B for all τ.
"""
function _multiplication_32!(temp_buffer::DensePropagator{T}, A::DensePropagator{T}, B::DenseOperator) where T<:Number
    _multiplication_32!(temp_buffer.data, A.data, B.data)
    return temp_buffer
end

"""
    _multiplication_13!(temp_buffer::DensePropagator{T}, A::AbstractVector, B::DensePropagator{T}) where T<:Number

In-place vector-propagator element-wise multiplication: temp_buffer[τ,:,:] = A[τ] * B[τ,:,:].
"""
function _multiplication_13!(temp_buffer::DensePropagator{T}, A::AbstractVector, B::DensePropagator{T}) where T<:Number
    _multiplication_13!(temp_buffer.data, A, B.data)
    return temp_buffer
end

"""
    _multiplication_33!(temp_buffer::DensePropagator{T}, A::DensePropagator{T}, B::DensePropagator{T}) where T<:Number

In-place propagator-propagator multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B[τ,:,:].
"""
function _multiplication_33!(temp_buffer::DensePropagator{T}, A::DensePropagator{T}, B::DensePropagator{T}) where T<:Number
    _multiplication_33!(temp_buffer.data, A.data, B.data)
    return temp_buffer
end

"""
    _trace_3(A::DensePropagator{T}) where T<:Number

Compute trace of each τ-slice: result[τ] = tr(A[τ,:,:]).
"""
function _trace_3(A::DensePropagator{T}) where T<:Number
    return _trace_3(A.data)
end

"""
    _trace_3!(trace_buffer::Vector{<:Real}, A::DensePropagator{<:Number})

In-place version: writes trace results to trace_buffer.
"""
function _trace_3!(trace_buffer::Vector{<:Real}, A::DensePropagator{<:Number})
    return _trace_3!(trace_buffer, A.data)
end
