"""
Linear algebra operations for BlockDenseArray (time-dependent block-dense arrays).

Implements:
- Array addition: A + B (flux must match)
- Scalar multiplication: α * A
- In-place operations: A .+= α .* B

These operations mirror the BlockDenseOperator algebra but work on 3D time-dependent blocks.
"""

using QMBToolkitDLR
using QMBToolkitED.BlockDense

"""
    Base.:+(A::BlockDenseArray{T}, B::BlockDenseArray{T}) where T -> BlockDenseArray{T}

Add two BlockDenseArray with the same flux.

# Arguments
- `A::BlockDenseArray{T}`: First array
- `B::BlockDenseArray{T}`: Second array

# Returns
- `BlockDenseArray{T}`: Sum array A+B

# Requirements
- Both arrays must have the same basis
- Both arrays must have the same flux
- Blocks are added element-wise, missing blocks treated as zero
"""
function Base.:+(A::BlockDenseArray{T}, B::BlockDenseArray{T}) where T
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

    return BlockDenseArray{T}(A.basis, A.flux, result_blocks)
end

"""
    Base.:-(A::BlockDenseArray{T}, B::BlockDenseArray{T}) where T -> BlockDenseArray{T}

Subtract two BlockDenseArray with the same flux.

Equivalent to A + (-1)*B.
"""
function Base.:-(A::BlockDenseArray{T}, B::BlockDenseArray{T}) where T
    return A + (-1) * B
end

"""
    Base.:*(α::Number, A::BlockDenseArray{T}) where T -> BlockDenseArray{T}

Multiply BlockDenseArray by scalar.

# Arguments
- `α::Number`: Scalar coefficient
- `A::BlockDenseArray{T}`: Array

# Returns
- `BlockDenseArray{T}`: Scaled array α*A with same flux

# Notes
- All blocks are scaled by α
- If α = 0, returns array with no blocks (empty dict)
"""
function Base.:*(α::Number, A::BlockDenseArray{T}) where T
    if iszero(α)
        # Return zero array (no blocks)
        return BlockDenseArray{T}(A.basis, A.flux, Dict{Tuple{Int,Int}, Array{T,3}}())
    end

    # Scale all blocks
    scaled_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, block) in A.blocks
        scaled_blocks[key] = α * block
    end

    return BlockDenseArray{T}(A.basis, A.flux, scaled_blocks)
end

"""
    Base.:*(A::BlockDenseArray{T}, α::Number) where T -> BlockDenseArray{T}

Multiply BlockDenseArray by scalar (commutative version).
"""
Base.:*(A::BlockDenseArray{T}, α::Number) where T = α * A

"""
    Base.:/(A::BlockDenseArray{T}, α::Number) where T -> BlockDenseArray{T}

Divide BlockDenseArray by scalar.

Equivalent to (1/α) * A.
"""
Base.:/(A::BlockDenseArray{T}, α::Number) where T = (1/α) * A

"""
    Base.:-(A::BlockDenseArray{T}) where T -> BlockDenseArray{T}

Unary negation of BlockDenseArray.

Equivalent to (-1) * A.
"""
Base.:-(A::BlockDenseArray{T}) where T = (-1) * A

"""
    Base.copyto!(dest::BlockDenseArray{T}, src::BlockDenseArray{T}) where T -> BlockDenseArray{T}

Copy contents of src into dest.

# Requirements
- dest and src must have same basis
- dest and src must have same flux
- dest must have blocks allocated for all keys in src

# Notes
- Only copies block contents, not structure
- dest must already have the correct block structure
"""
function Base.copyto!(dest::BlockDenseArray{T}, src::BlockDenseArray{T}) where T
    @assert dest.basis === src.basis "Arrays must have same basis"
    @assert dest.flux.delta_qn == src.flux.delta_qn "Arrays must have same flux"

    for (key, src_block) in src.blocks
        @assert haskey(dest.blocks, key) "Destination missing block $key"
        copyto!(dest.blocks[key], src_block)
    end

    return dest
end

"""
    Base.fill!(A::BlockDenseArray{T}, value) where T -> BlockDenseArray{T}

Fill all elements of all blocks with a value.

# Arguments
- `A::BlockDenseArray{T}`: Array to fill
- `value`: Value to fill with (typically 0)

# Returns
- `A`: The input array (modified)
"""
function Base.fill!(A::BlockDenseArray{T}, value) where T
    for (key, block) in A.blocks
        fill!(block, value)
    end
    return A
end

"""
    add_scaled!(dest::BlockDenseArray{T}, α::Number, src::BlockDenseArray{T}) where T

In-place operation: dest .+= α .* src

Equivalent to dest = dest + α * src but more efficient (avoids allocations).

# Arguments
- `dest::BlockDenseArray{T}`: Destination array (modified)
- `α::Number`: Scalar coefficient
- `src::BlockDenseArray{T}`: Source array

# Requirements
- dest and src must have same basis
- dest and src must have same flux
- dest must have blocks for all keys in src
"""
function add_scaled!(dest::BlockDenseArray{T}, α::Number, src::BlockDenseArray{T}) where T
    @assert dest.basis === src.basis "Arrays must have same basis"
    @assert dest.flux.delta_qn == src.flux.delta_qn "Arrays must have same flux"

    if iszero(α)
        return dest
    end

    for (key, src_block) in src.blocks
        if haskey(dest.blocks, key)
            # dest_block .+= α .* src_block
            dest.blocks[key] .+= α .* src_block
        else
            # This block doesn't exist in dest yet - need to add it
            dest.blocks[key] = α * src_block
        end
    end

    return dest
end

"""
    Base.zeros(::Type{BlockDenseArray{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where T

Create a zero BlockDenseArray with specified basis, flux, and time dimension.

Allocates zero blocks for all sector pairs allowed by the flux.

# Arguments
- `::Type{BlockDenseArray{T}}`: Type parameter
- `basis::BlockDenseBasis`: Basis defining block structure
- `flux::OperatorFlux`: Quantum number flux
- `n_tau::Int`: Number of time points

# Returns
- `BlockDenseArray{T}`: Zero array with appropriate block structure
"""
function Base.zeros(::Type{BlockDenseArray{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where T
    blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    # For now, allocate diagonal blocks (flux = 0 case)
    # In full implementation, would determine blocks from flux
    if flux.delta_qn == zero_qn(typeof(basis.sectors[1].qn))
        # Diagonal blocks only
        for (i, sector) in enumerate(basis.sectors)
            dim = sector.dim
            blocks[(i, i)] = zeros(T, n_tau, dim, dim)
        end
    else
        # For non-zero flux, need to determine allowed blocks
        # This would require quantum number arithmetic
        # For now, just create empty dict
    end

    return BlockDenseArray{T}(basis, flux, blocks)
end
