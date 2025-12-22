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
struct BlockDensePropagator{T<:Number}
    basis::BlockDenseBasis
    flux::OperatorFlux
    blocks::Dict{Tuple{Int,Int}, Array{T,3}}
end

# Constructor
function BlockDensePropagator(basis::BlockDenseBasis, flux::OperatorFlux, blocks::Dict{Tuple{Int,Int}, Array{T,3}}) where T
    return BlockDensePropagator{T}(basis, flux, blocks)
end

"""
    Base.zeros(::Type{BlockDensePropagator{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where T

Create a zero BlockDensePropagator with specified basis, flux, and time dimension.

Allocates zero blocks for all sector pairs allowed by the flux.
"""
function Base.zeros(::Type{BlockDensePropagator{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where T
    blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

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
    Base.:+(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T

Add two BlockDensePropagator with the same flux.
"""
function Base.:+(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T
    @assert A.basis === B.basis "Arrays must have same basis"
    @assert A.flux.delta_qn == B.flux.delta_qn "Arrays must have same flux to be added"

    # Start with copy of A's blocks
    result_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, A_block) in A.blocks
        result_blocks[key] = copy(A_block)
    end

    # Add B's blocks
    for (key, B_block) in B.blocks
        if haskey(result_blocks, key)
            result_blocks[key] .+= B_block
        else
            result_blocks[key] = copy(B_block)
        end
    end

    return BlockDensePropagator{T}(A.basis, A.flux, result_blocks)
end

"""
    Base.:-(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T

Subtract two BlockDensePropagator with the same flux.
"""
function Base.:-(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T
    return A + (-1) * B
end

"""
    Base.:*(alpha::Number, A::BlockDensePropagator{T}) where T

Multiply BlockDensePropagator by scalar.
"""
function Base.:*(alpha::Number, A::BlockDensePropagator{T}) where T
    if iszero(alpha)
        # Return zero propagator (no blocks)
        return BlockDensePropagator{T}(A.basis, A.flux, Dict{Tuple{Int,Int}, Array{T,3}}())
    end

    # Scale all blocks
    scaled_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, block) in A.blocks
        scaled_blocks[key] = alpha * block
    end

    return BlockDensePropagator{T}(A.basis, A.flux, scaled_blocks)
end

Base.:*(A::BlockDensePropagator{T}, alpha::Number) where T = alpha * A
Base.:/(A::BlockDensePropagator{T}, alpha::Number) where T = (1/alpha) * A
Base.:-(A::BlockDensePropagator{T}) where T = (-1) * A

"""
    Base.copyto!(dest::BlockDensePropagator{T}, src::BlockDensePropagator{T}) where T

Copy contents of src into dest.
"""
function Base.copyto!(dest::BlockDensePropagator{T}, src::BlockDensePropagator{T}) where T
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
function Base.fill!(A::BlockDensePropagator{T}, value) where T
    for (key, block) in A.blocks
        fill!(block, value)
    end
    return A
end

"""
    add_scaled!(dest::BlockDensePropagator{T}, alpha::Number, src::BlockDensePropagator{T}) where T

In-place operation: dest .+= alpha .* src
"""
function add_scaled!(dest::BlockDensePropagator{T}, alpha::Number, src::BlockDensePropagator{T}) where T
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
mutable struct BlockDenseDLRPropagator{T<:Number}
    vals::BlockDensePropagator{T}
    coeffs::BlockDensePropagator{T}
end

# Constructors
function BlockDenseDLRPropagator(vals::BlockDensePropagator{T}, coeffs::BlockDensePropagator{T}) where T
    return BlockDenseDLRPropagator{T}(vals, coeffs)
end

function BlockDenseDLRPropagator{T}(basis::BlockDenseBasis, flux::OperatorFlux, r::Int) where T
    vals = zeros(BlockDensePropagator{T}, basis, flux, r)
    coeffs = zeros(BlockDensePropagator{T}, basis, flux, r)
    return BlockDenseDLRPropagator{T}(vals, coeffs)
end

function BlockDenseDLRPropagator(vals::BlockDensePropagator{T}, ::Nothing) where T
    # Get the first block to determine r dimension
    first_block = first(values(vals.blocks))
    r = size(first_block, 1)
    coeffs = zeros(BlockDensePropagator{T}, vals.basis, vals.flux, r)
    return BlockDenseDLRPropagator{T}(vals, coeffs)
end

function BlockDenseDLRPropagator(::Nothing, coeffs::BlockDensePropagator{T}) where T
    # Get the first block to determine r dimension
    first_block = first(values(coeffs.blocks))
    r = size(first_block, 1)
    vals = zeros(BlockDensePropagator{T}, coeffs.basis, coeffs.flux, r)
    return BlockDenseDLRPropagator{T}(vals, coeffs)
end

