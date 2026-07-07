"""
GreenFunction - Green's function storage types.

The Green's function is a physical observable in small orbital space,
stored as plain 3D arrays (n_orb1, n_orb2, n_tau).
"""

"""
    GreenDLR{T<:Number}

DLR representation of a Green's function.

Contains both the values on the DLR tau grid and the DLR coefficients
as plain 3D arrays.

Fields:

  - `vals::Array{T, 3}`: Values on the DLR tau grid, shape (n_orb, n_orb, r)
  - `coeffs::Array{T, 3}`: DLR expansion coefficients, shape (n_orb, n_orb, r)
"""
mutable struct GreenDLR{T <: Number}
    vals::Array{T, 3}
    coeffs::Array{T, 3}
end

# Constructors
function GreenDLR{T}(r::Int, dim1::Int, dim2::Int) where {T}
    vals = zeros(T, dim1, dim2, r)
    coeffs = zeros(T, dim1, dim2, r)
    return GreenDLR{T}(vals, coeffs)
end

#=============================================================================
# MPI Operations
=============================================================================#

"""
    mpi_reduce_and_add!(green::GreenDLR{T}, local_result::Array{T,3}, comm::MPI.Comm) where T

Perform MPI Allreduce on local_result and add to green.vals.
"""
function mpi_reduce_and_add!(green::GreenDLR{T},
        local_result::Array{T, 3}, comm::MPI.Comm) where {T}
    order_result = MPI.Allreduce(local_result, MPI.SUM, comm)
    green.vals .+= order_result
    return green
end
