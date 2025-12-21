"""
DensePropagator - time-dependent dense propagator wrapper.

Provides a wrapper around Array{T,3} for consistent interface with BlockDensePropagator.
"""

"""
    DensePropagator{T<:Number}

Time-dependent dense propagator. A wrapper around Array{T,3} where:
- First dimension: tau grid points
- Second dimension: row index (Fock space dimension)
- Third dimension: column index (Fock space dimension)

This is a trivial wrapper that provides consistent interface with BlockDensePropagator.
"""
struct DensePropagator{T<:Number}
    data::Array{T,3}
end

# Constructors
DensePropagator(data::Array{T,3}) where T = DensePropagator{T}(data)

function Base.zeros(::Type{DensePropagator{T}}, n_tau::Int, dim1::Int, dim2::Int) where T
    return DensePropagator{T}(zeros(T, n_tau, dim1, dim2))
end

# Size and access
Base.size(p::DensePropagator) = size(p.data)
Base.size(p::DensePropagator, d::Int) = size(p.data, d)
Base.getindex(p::DensePropagator, args...) = getindex(p.data, args...)
Base.setindex!(p::DensePropagator, val, args...) = setindex!(p.data, val, args...)
Base.view(p::DensePropagator, args...) = view(p.data, args...)

# Arithmetic
function Base.:+(A::DensePropagator{T}, B::DensePropagator{T}) where T
    return DensePropagator{T}(A.data .+ B.data)
end

function Base.:-(A::DensePropagator{T}, B::DensePropagator{T}) where T
    return DensePropagator{T}(A.data .- B.data)
end

function Base.:*(alpha::Number, A::DensePropagator{T}) where T
    return DensePropagator{T}(alpha .* A.data)
end

Base.:*(A::DensePropagator{T}, alpha::Number) where T = alpha * A
Base.:/(A::DensePropagator{T}, alpha::Number) where T = (1/alpha) * A
Base.:-(A::DensePropagator{T}) where T = (-1) * A

function Base.copyto!(dest::DensePropagator{T}, src::DensePropagator{T}) where T
    copyto!(dest.data, src.data)
    return dest
end

function Base.fill!(A::DensePropagator{T}, value) where T
    fill!(A.data, value)
    return A
end

#=============================================================================
# DenseDLRPropagator - DLR representation wrapper
=============================================================================#

"""
    DenseDLRPropagator{T<:Number}

DLR representation of a dense propagator.

Contains both the values on the DLR tau grid and the DLR coefficients,
both wrapped as DensePropagator{T}.

Fields:
- `vals::DensePropagator{T}`: Values on the DLR tau grid
- `coeffs::DensePropagator{T}`: DLR expansion coefficients
"""
mutable struct DenseDLRPropagator{T<:Number}
    vals::DensePropagator{T}
    coeffs::DensePropagator{T}
end

# Constructors
function DenseDLRPropagator(vals::DensePropagator{T}, coeffs::DensePropagator{T}) where T
    return DenseDLRPropagator{T}(vals, coeffs)
end

function DenseDLRPropagator{T}(r::Int, dim1::Int, dim2::Int) where T
    vals = zeros(DensePropagator{T}, r, dim1, dim2)
    coeffs = zeros(DensePropagator{T}, r, dim1, dim2)
    return DenseDLRPropagator{T}(vals, coeffs)
end

function DenseDLRPropagator(vals::DensePropagator{T}, ::Nothing) where T
    coeffs = zeros(DensePropagator{T}, size(vals)...)
    return DenseDLRPropagator{T}(vals, coeffs)
end

function DenseDLRPropagator(::Nothing, coeffs::DensePropagator{T}) where T
    vals = zeros(DensePropagator{T}, size(coeffs)...)
    return DenseDLRPropagator{T}(vals, coeffs)
end

# Convenience constructor that infers type parameter
DenseDLRPropagator(r::Int, dim1::Int, dim2::Int) = DenseDLRPropagator{Float64}(r, dim1, dim2)
