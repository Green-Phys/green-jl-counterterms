"""
BlockDensePropagator - block-sparse time-dependent propagator.

Provides a block-sparse wrapper exploiting quantum number conservation for
time-dependent propagator data.
"""

using QMBToolkitED.BlockDense

"""
    BlockDensePropagator{T<:Number}

Time-dependent block-dense propagator for DLR operations.

Each block is stored as a 3D array: (time_points, row_dim, col_dim)
where time_points is either tau grid or DLR grid.

Fields:
- `basis::BlockDenseBasis`: Basis defining the block structure
- `flux::OperatorFlux`: Quantum number change (flux) of the operator
- `blocks::Dict{Tuple{Int,Int}, Array{T,3}}`: Blocks indexed by (target_sector, source_sector)
"""
struct BlockDensePropagator{T <: Number}
    basis::BlockDenseBasis
    flux::OperatorFlux
    blocks::Dict{Tuple{Int, Int}, Array{T, 3}}
end

# Constructor
function BlockDensePropagator(basis::BlockDenseBasis, flux::OperatorFlux,
        blocks::Dict{Tuple{Int, Int}, Array{T, 3}},) where {T}
    return BlockDensePropagator{T}(basis, flux, blocks)
end

"""
    Base.zeros(::Type{BlockDensePropagator{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where T

Create a zero BlockDensePropagator with specified basis, flux, and time dimension.

Allocates zero blocks for all sector pairs allowed by the flux.
"""
function Base.zeros(::Type{BlockDensePropagator{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where {T}
    blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()

    # For zero flux, allocate diagonal blocks only
    if flux.delta_qn == zero_qn(typeof(basis.sectors[1].qn))
        # Diagonal blocks only
        for (i, sector) in enumerate(basis.sectors)
            dim = sector.dim
            blocks[(i, i)] = zeros(T, n_tau, dim, dim)
        end
    else
        # For non-zero flux, need to determine allowed blocks based on quantum number arithmetic
        # Currently just create empty dict - will be populated as needed
    end

    return BlockDensePropagator{T}(basis, flux, blocks)
end

#=============================================================================
# BlockDensePropagator Arithmetic Operations
=============================================================================#

"""
    Base.:*(alpha::Number, A::BlockDensePropagator{T}) where T

Multiply BlockDensePropagator by scalar.
"""
function Base.:*(alpha::Number, A::BlockDensePropagator{T}) where {T}
    if iszero(alpha)
        # Return zero propagator (no blocks)
        return BlockDensePropagator{T}(A.basis, A.flux, Dict{Tuple{Int, Int}, Array{T, 3}}())
    end

    # Scale all blocks
    scaled_blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    for (key, block) in A.blocks
        scaled_blocks[key] = alpha * block
    end

    return BlockDensePropagator{T}(A.basis, A.flux, scaled_blocks)
end

Base.:*(A::BlockDensePropagator{T}, alpha::Number) where {T} = alpha * A

"""
    Base.copyto!(dest::BlockDensePropagator{T}, src::BlockDensePropagator{T}) where T

Copy contents of src into dest.
"""
function Base.copyto!(dest::BlockDensePropagator{T}, src::BlockDensePropagator{T}) where {T}
    @assert dest.basis === src.basis "Arrays must have same basis"
    @assert dest.flux.delta_qn == src.flux.delta_qn "Arrays must have same flux"

    for (key, src_block) in src.blocks
        @assert haskey(dest.blocks, key) "Destination missing block $key"
        copyto!(dest.blocks[key], src_block)
    end

    return dest
end

"""
    Base.fill!(A::BlockDensePropagator{T}, value) where T

Fill all elements of all blocks with a value.
"""
function Base.fill!(A::BlockDensePropagator{T}, value) where {T}
    for (key, block) in A.blocks
        fill!(block, value)
    end
    return A
end

"""
    add_scaled!(dest::BlockDensePropagator{T}, alpha::Number, src::BlockDensePropagator{T}) where T

In-place operation: dest .+= alpha .* src
"""
function add_scaled!(dest::BlockDensePropagator{T}, alpha::Number, src::BlockDensePropagator{T}) where {T}
    @assert dest.basis === src.basis "Arrays must have same basis"
    @assert dest.flux.delta_qn == src.flux.delta_qn "Arrays must have same flux"

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
# BlockDenseDLRPropagator - DLR representation wrapper
=============================================================================#

"""
    BlockDenseDLRPropagator{T<:Number}

DLR representation of a block-dense propagator.

Contains both the values on the DLR tau grid and the DLR coefficients,
both wrapped as BlockDensePropagator{T}.

Fields:
- `vals::BlockDensePropagator{T}`: Values on the DLR tau grid
- `coeffs::BlockDensePropagator{T}`: DLR expansion coefficients
"""
mutable struct BlockDenseDLRPropagator{T <: Number}
    vals::BlockDensePropagator{T}
    coeffs::BlockDensePropagator{T}
end

# Constructors
function BlockDenseDLRPropagator(vals::BlockDensePropagator{T}, coeffs::BlockDensePropagator{T}) where {T}
    return BlockDenseDLRPropagator{T}(vals, coeffs)
end

function BlockDenseDLRPropagator{T}(basis::BlockDenseBasis, flux::OperatorFlux, r::Int) where {T}
    vals = zeros(BlockDensePropagator{T}, basis, flux, r)
    coeffs = zeros(BlockDensePropagator{T}, basis, flux, r)
    return BlockDenseDLRPropagator{T}(vals, coeffs)
end

#=============================================================================
# Interface Implementations
=============================================================================#

"""
    zeros_like(template::BlockDensePropagator{T}) where T

Create a zero-initialized BlockDensePropagator with the same structure as template.
"""
function zeros_like(template::BlockDensePropagator{T}) where {T}
    new_blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    for (key, block) in template.blocks
        new_blocks[key] = zeros(T, size(block)...)
    end
    return BlockDensePropagator{T}(template.basis, template.flux, new_blocks)
end

"""
    get_data(prop::BlockDensePropagator)

Get the underlying blocks dictionary.
"""
get_data(prop::BlockDensePropagator) = prop.blocks

#=============================================================================
# MPI Operations
=============================================================================#

"""
    create_local_accumulator(::Type{BlockDensePropagator{T}}, template::BlockDensePropagator{T}) where T

Create a zero-initialized local accumulator for MPI reduction.
Returns a BlockDensePropagator with same structure as template.
"""
function create_local_accumulator(::Type{BlockDensePropagator{T}}, template::BlockDensePropagator{T}) where {T}
    return zeros_like(template)
end

"""
    accumulate_contribution!(accumulator::BlockDensePropagator{T}, prefactor, contribution::BlockDensePropagator{T}) where T

Add scaled contribution to accumulator: accumulator += prefactor * contribution
"""
function accumulate_contribution!(accumulator::BlockDensePropagator{T}, prefactor,
        contribution::BlockDensePropagator{T},) where {T}
    add_scaled!(accumulator, prefactor, contribution)
    return accumulator
end

"""
    mpi_reduce_and_add!(propagator::BlockDenseDLRPropagator{T}, local_result::BlockDensePropagator{T}, comm::MPI.Comm) where T

Perform MPI Allreduce on local_result block-by-block and add to propagator.vals.
"""
function mpi_reduce_and_add!(propagator::BlockDenseDLRPropagator{T},
        local_result::BlockDensePropagator{T}, comm::MPI.Comm,) where {T}
    for (key, local_block) in local_result.blocks
        # Reduce this block across all ranks
        global_block = MPI.Allreduce(local_block, MPI.SUM, comm)

        # Add to propagator (create block if it doesn't exist)
        if haskey(propagator.vals.blocks, key)
            propagator.vals.blocks[key] .+= global_block
        else
            propagator.vals.blocks[key] = copy(global_block)
        end
    end
    return propagator
end
