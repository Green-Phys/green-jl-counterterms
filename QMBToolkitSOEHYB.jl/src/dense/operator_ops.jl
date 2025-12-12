"""
Matrix multiplication operations for dense Array{T,3} representation.

These functions perform various matrix multiplications along the τ-dimension:
- _multiplication_23: Matrix-tensor multiplication (2D matrix × 3D tensor)
- _multiplication_32: Tensor-matrix multiplication (3D tensor × 2D matrix)
- _multiplication_13: Vector-tensor multiplication (1D vector × 3D tensor, element-wise)
- _multiplication_33: Tensor-tensor multiplication (3D tensor × 3D tensor)
- _trace_3: Trace operation over τ-slices of 3D tensor

Each operation has both in-place (!) and allocating versions.
"""

using LinearAlgebra

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
