"""
Propagator - block-sparse time-dependent propagator.

Exploits quantum number conservation for efficient block-sparse storage
of time-dependent propagator data.
"""

using QMBToolkitED: Basis, QuantumNumber

"""
    Propagator{T<:Number}

Time-dependent block-sparse propagator for DLR operations.

Each block is stored as a 3D array: (row_dim, col_dim, time_points)
where time_points is either tau grid or DLR grid.
This layout ensures contiguous memory access for matrix operations at each time point.

Fields:

  - `basis::Basis`: Basis defining the block structure
  - `flux::QuantumNumber`: Quantum number change (flux) of the operator
  - `blocks::Dict{Tuple{Int,Int}, Array{T,3}}`: Blocks indexed by (target_sector, source_sector)
"""
struct Propagator{T <: Number}
    basis::Basis
    flux::QuantumNumber
    blocks::Dict{Tuple{Int, Int}, Array{T, 3}}
end

# Constructor
function Propagator(basis::Basis, flux::QuantumNumber,
        blocks::Dict{Tuple{Int, Int}, Array{T, 3}}) where {T}
    return Propagator{T}(basis, flux, blocks)
end

"""
    Base.zeros(::Type{Propagator{T}}, basis::Basis, flux::QuantumNumber, n_tau::Int) where T

Create a zero Propagator with specified basis, flux, and time dimension.

Allocates zero blocks for all sector pairs allowed by the flux.
"""
function Base.zeros(::Type{Propagator{T}}, basis::Basis, flux::QuantumNumber, n_tau::Int) where {T}
    blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()

    # For zero flux, allocate diagonal blocks only
    if flux == zero(typeof(basis.sectors[1].qn))
        for (i, sector) in enumerate(basis.sectors)
            dim = sector.dim
            blocks[(i, i)] = zeros(T, dim, dim, n_tau)
        end
    else
        # For non-zero flux, allocate blocks where sector_i.qn + flux == sector_j.qn
        for (i, sector_i) in enumerate(basis.sectors)
            target_qn = sector_i.qn + flux
            for (j, sector_j) in enumerate(basis.sectors)
                if sector_j.qn == target_qn
                    blocks[(i, j)] = zeros(T, sector_i.dim, sector_j.dim, n_tau)
                end
            end
        end
    end

    return Propagator{T}(basis, flux, blocks)
end

#=============================================================================
# Propagator Arithmetic Operations
=============================================================================#

"""
    Base.:*(alpha::Number, A::Propagator{T}) where T

Multiply Propagator by scalar.
"""
function Base.:*(alpha::Number, A::Propagator{T}) where {T}
    if iszero(alpha)
        return Propagator{T}(A.basis, A.flux, Dict{Tuple{Int, Int}, Array{T, 3}}())
    end

    scaled_blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    for (key, block) in A.blocks
        scaled_blocks[key] = alpha * block
    end

    return Propagator{T}(A.basis, A.flux, scaled_blocks)
end

Base.:*(A::Propagator{T}, alpha::Number) where {T} = alpha * A

"""
    Base.copyto!(dest::Propagator{T}, src::Propagator{T}) where T

Copy contents of src into dest.
"""
function Base.copyto!(dest::Propagator{T}, src::Propagator{T}) where {T}
    @boundscheck @assert dest.basis === src.basis "Arrays must have same basis"
    @boundscheck @assert dest.flux == src.flux "Arrays must have same flux"

    for (key, src_block) in src.blocks
        @boundscheck @assert haskey(dest.blocks, key) "Destination missing block $key"
        copyto!(dest.blocks[key], src_block)
    end

    return dest
end

"""
    Base.fill!(A::Propagator{T}, value) where T

Fill all elements of all blocks with a value.
"""
function Base.fill!(A::Propagator{T}, value) where {T}
    for (key, block) in A.blocks
        fill!(block, value)
    end
    return A
end

"""
    add_scaled!(dest::Propagator{T}, alpha::Number, src::Propagator{T}) where T

In-place operation: dest .+= alpha .* src
"""
function add_scaled!(dest::Propagator{T}, alpha::Number, src::Propagator{T}) where {T}
    @boundscheck @assert dest.basis === src.basis "Arrays must have same basis"
    @boundscheck @assert dest.flux == src.flux "Arrays must have same flux"

    if iszero(alpha)
        return dest
    end

    for (key, src_block) in src.blocks
        if haskey(dest.blocks, key)
            dest.blocks[key] .+= alpha .* src_block
        else
            dest.blocks[key] = alpha * src_block
        end
    end

    return dest
end

#=============================================================================
# DLRPropagator - DLR representation wrapper
=============================================================================#

"""
    DLRPropagator{T<:Number}

DLR representation of a block-sparse propagator.

Contains both the values on the DLR tau grid and the DLR coefficients,
both wrapped as Propagator{T}.

Fields:

  - `vals::Propagator{T}`: Values on the DLR tau grid
  - `coeffs::Propagator{T}`: DLR expansion coefficients
"""
mutable struct DLRPropagator{T <: Number}
    vals::Propagator{T}
    coeffs::Propagator{T}
end

# Constructors
function DLRPropagator(vals::Propagator{T}, coeffs::Propagator{T}) where {T}
    return DLRPropagator{T}(vals, coeffs)
end

function DLRPropagator{T}(basis::Basis, flux::QuantumNumber, r::Int) where {T}
    vals = zeros(Propagator{T}, basis, flux, r)
    coeffs = zeros(Propagator{T}, basis, flux, r)
    return DLRPropagator{T}(vals, coeffs)
end

#=============================================================================
# Interface Implementations
=============================================================================#

"""
    zeros_like(template::Propagator{T}) where T

Create a zero-initialized Propagator with the same structure as template.
"""
function zeros_like(template::Propagator{T}) where {T}
    new_blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    for (key, block) in template.blocks
        new_blocks[key] = zeros(T, size(block)...)
    end
    return Propagator{T}(template.basis, template.flux, new_blocks)
end

#=============================================================================
# MPI Operations
=============================================================================#

"""
    mpi_reduce_and_add!(propagator::DLRPropagator{T}, local_result::Propagator{T}, comm::MPI.Comm) where T

Perform MPI Allreduce on local_result block-by-block and add to propagator.vals.
"""
function mpi_reduce_and_add!(propagator::DLRPropagator{T},
        local_result::Propagator{T}, comm::MPI.Comm) where {T}
    for (key, local_block) in local_result.blocks
        global_block = MPI.Allreduce(local_block, MPI.SUM, comm)

        if haskey(propagator.vals.blocks, key)
            propagator.vals.blocks[key] .+= global_block
        else
            propagator.vals.blocks[key] = copy(global_block)
        end
    end
    return propagator
end
