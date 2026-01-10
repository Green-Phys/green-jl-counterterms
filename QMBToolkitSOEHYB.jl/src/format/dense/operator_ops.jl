"""
Matrix multiplication operations for dense representation.

These functions perform various matrix multiplications along the τ-dimension:
- _multiplication_23_dense!: Operator-propagator multiplication (DenseOperator × DensePropagator)
- _multiplication_32_dense!: Propagator-operator multiplication (DensePropagator × DenseOperator)
- _multiplication_13_dense!: Vector-propagator multiplication (1D vector × DensePropagator, element-wise)
- _multiplication_33_dense!: Propagator-propagator multiplication (DensePropagator × DensePropagator)
- _trace_3_dense!: Trace operation over τ-slices of DensePropagator

All operations are in-place for efficiency.
Works with both wrapper types (DenseOperator, DensePropagator) and raw arrays.
"""

using LinearAlgebra
using QMBToolkitED.Dense: DenseOperator

# ============================================================================
# Matrix-Tensor Multiplications: (τ, i, j) shaped arrays
# ============================================================================

"""
    _multiplication_23_dense!(temp_buffer::Array{T,3}, A::Array{T,2}, B::Array{T,3}) where T<:Number

In-place matrix-tensor multiplication: temp_buffer[τ,:,:] = A * B[τ,:,:] for all τ.
Writes result to temp_buffer. Caller should copy back to input if needed.
"""
function _multiplication_23_dense!(temp_buffer::Array{T, 3}, A::Array{T, 2}, B::Array{
        T, 3,}) where {T <: Number}
    n_tau = size(B, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer,k,:,:), A, view(B,k,:,:))
    end
    return temp_buffer
end

"""
    _multiplication_32_dense!(temp_buffer::Array{T,3}, A::Array{T,3}, B::Array{T,2}) where T<:Number

In-place tensor-matrix multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B for all τ.
"""
function _multiplication_32_dense!(temp_buffer::Array{T, 3}, A::Array{T, 3}, B::Array{
        T, 2,}) where {T <: Number}
    n_tau = size(A, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer,k,:,:), view(A,k,:,:), B)
    end
    return temp_buffer
end

"""
    _multiplication_13_dense!(temp_buffer::Array{T,3}, A::AbstractVector, B::Array{T,3}) where T<:Number

In-place vector-tensor element-wise multiplication: temp_buffer[τ,:,:] = A[τ] * B[τ,:,:].
"""
function _multiplication_13_dense!(temp_buffer::Array{T, 3}, A::AbstractVector, B::Array{
        T, 3,}) where {T <: Number}
    n_tau = size(B, 1)
    @inbounds for k in 1:n_tau
        @views temp_buffer[k, :, :] .= A[k] .* B[k, :, :]
    end
    return temp_buffer
end

"""
    _multiplication_33_dense!(temp_buffer::Array{T,3}, A::Array{T,3}, B::Array{T,3}) where T<:Number

In-place tensor-tensor multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B[τ,:,:].
"""
function _multiplication_33_dense!(temp_buffer::Array{T, 3}, A::Array{T, 3}, B::Array{
        T, 3,}) where {T <: Number}
    n_tau = size(A, 1)
    @inbounds for k in 1:n_tau
        mul!(view(temp_buffer,k,:,:), view(A,k,:,:), view(B,k,:,:))
    end
    return temp_buffer
end

# ============================================================================
# Trace Operations
# ============================================================================

"""
    _trace_3_dense!(trace_buffer::Vector{<:Real}, A::Array{<:Number,3})

In-place version: writes trace results to trace_buffer.
"""
function _trace_3_dense!(trace_buffer::Vector{<:Real}, A::Array{<:Number, 3})
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    o1 == o2 || error("Matrix to be traced must be square: got dimensions $o1 x $o2")
    @inbounds for k in 1:n_tau
        trace_buffer[k] = tr(@view A[k, :, :])
    end
    return trace_buffer
end

# ============================================================================
# Wrapper Methods for DenseOperator and DensePropagator
# ============================================================================

"""
    _multiplication_23_dense!(temp_buffer::DensePropagator{T}, A::DenseOperator, B::DensePropagator{T}) where T<:Number

In-place operator-propagator multiplication: temp_buffer[τ,:,:] = A * B[τ,:,:] for all τ.
"""
function _multiplication_23_dense!(temp_buffer::DensePropagator{T}, A::DenseOperator,
        B::DensePropagator{T}) where {T <: Number}
    _multiplication_23_dense!(temp_buffer.data, A.data, B.data)
    return temp_buffer
end

"""
    _multiplication_32_dense!(temp_buffer::DensePropagator{T}, A::DensePropagator{T}, B::DenseOperator) where T<:Number

In-place propagator-operator multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B for all τ.
"""
function _multiplication_32_dense!(temp_buffer::DensePropagator{T}, A::DensePropagator{T},
        B::DenseOperator) where {T <: Number}
    _multiplication_32_dense!(temp_buffer.data, A.data, B.data)
    return temp_buffer
end

"""
    _multiplication_13_dense!(temp_buffer::DensePropagator{T}, A::AbstractVector, B::DensePropagator{T}) where T<:Number

In-place vector-propagator element-wise multiplication: temp_buffer[τ,:,:] = A[τ] * B[τ,:,:].
"""
function _multiplication_13_dense!(temp_buffer::DensePropagator{T}, A::AbstractVector,
        B::DensePropagator{T}) where {T <: Number}
    _multiplication_13_dense!(temp_buffer.data, A, B.data)
    return temp_buffer
end

"""
    _multiplication_33_dense!(temp_buffer::DensePropagator{T}, A::DensePropagator{T}, B::DensePropagator{T}) where T<:Number

In-place propagator-propagator multiplication: temp_buffer[τ,:,:] = A[τ,:,:] * B[τ,:,:].
"""
function _multiplication_33_dense!(temp_buffer::DensePropagator{T}, A::DensePropagator{T},
        B::DensePropagator{T}) where {T <: Number}
    _multiplication_33_dense!(temp_buffer.data, A.data, B.data)
    return temp_buffer
end

"""
    _trace_3_dense!(trace_buffer::Vector{<:Real}, A::DensePropagator{<:Number})

In-place version: writes trace results to trace_buffer.
"""
function _trace_3_dense!(trace_buffer::Vector{<:Real}, A::DensePropagator{<:Number})
    return _trace_3_dense!(trace_buffer, A.data)
end

# ============================================================================
# Interface Implementations
# ============================================================================

# Operator-propagator multiplication: result = op * prop
function multiply_op_prop!(result::DensePropagator{T}, op::DenseOperator, prop::DensePropagator{T}) where {T}
    _multiplication_23_dense!(result, op, prop)
end

# Propagator-operator multiplication: result = prop * op
function multiply_prop_op!(result::DensePropagator{T}, prop::DensePropagator{T}, op::DenseOperator) where {T}
    _multiplication_32_dense!(result, prop, op)
end

# Propagator-propagator multiplication: result = A * B
function multiply_prop_prop!(result::DensePropagator{T}, A::DensePropagator{T}, B::DensePropagator{T}) where {T}
    _multiplication_33_dense!(result, A, B)
end

# Trace over spatial indices
function trace_prop!(result_buffer::Vector{<:Real}, prop::DensePropagator)
    _trace_3_dense!(result_buffer, prop)
end

#=============================================================================
# Product Buffer Allocation Helpers (Dense format - simple pre-allocation)
=============================================================================#

"""
    allocate_product_buffer(op::DenseOperator, prop::DensePropagator{T}, n_tau::Int) where T

Allocate a buffer for the result of op * prop.
For dense format, just allocate a zero array of appropriate size.
"""
function allocate_product_buffer(op::DenseOperator, prop::DensePropagator{T}, n_tau::Int) where {T}
    return zeros(DensePropagator{T}, n_tau, size(op.data, 1), size(prop.data, 3))
end

"""
    allocate_product_buffer(prop::DensePropagator{T}, op::DenseOperator, n_tau::Int) where T

Allocate a buffer for the result of prop * op.
"""
function allocate_product_buffer(prop::DensePropagator{T}, op::DenseOperator, n_tau::Int) where {T}
    return zeros(DensePropagator{T}, n_tau, size(prop.data, 2), size(op.data, 2))
end

"""
    allocate_product_buffer(A::DensePropagator{T}, B::DensePropagator{T}, n_tau::Int) where T

Allocate a buffer for the result of A * B.
"""
function allocate_product_buffer(A::DensePropagator{T}, B::DensePropagator{T}, n_tau::Int) where {T}
    return zeros(DensePropagator{T}, n_tau, size(A.data, 2), size(B.data, 3))
end
