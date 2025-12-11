"""
Linear algebra operations for block-dense operators.

Implements:
- Operator multiplication: A * B (block chaining with flux composition)
- Operator addition: A + B (flux must match)
- Scalar multiplication: α * A
- Adjoint: A' (flux reversal and block transposition)
"""

"""
    Base.:*(A::BlockDenseOperator, B::BlockDenseOperator) -> BlockDenseOperator

Multiply two block-dense operators by chaining blocks.

The resulting operator has flux = combine_qn(A.flux, B.flux).
Block (target, source) in the result is computed by summing over intermediate sectors:
    result[target, source] = Σ_mid A.blocks[target, mid] * B.blocks[mid, source]

Arguments:
- `A::BlockDenseOperator`: Left operator
- `B::BlockDenseOperator`: Right operator

Returns:
- `BlockDenseOperator`: Product operator A*B

Requirements:
- Both operators must act on the same basis
- For efficiency, only non-zero blocks are computed

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blockdense(basis, 1)
c1_dag = c1'

# Number operator: n1 = c1† * c1
n1 = c1_dag * c1
# flux(n1) = QN_N(1) + QN_N(-1) = QN_N(0)
```
"""
function Base.:*(A::BlockDenseOperator, B::BlockDenseOperator)
    @assert A.basis === B.basis "Operators must act on same basis"

    # Result flux is composition of input fluxes
    result_flux = OperatorFlux(combine_qn(A.flux.delta_qn, B.flux.delta_qn))

    # Storage for result blocks
    result_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    # Index B blocks by their "target" (mid) sector for O(1) lookup
    # B.blocks keys are (mid, src). We want to look up by 'mid'.
    B_by_mid = Dict{Int, Vector{Tuple{Int, Matrix{ComplexF64}}}}()
    for ((mid, src), B_block) in B.blocks
        if !haskey(B_by_mid, mid)
            B_by_mid[mid] = Tuple{Int, Matrix{ComplexF64}}[]
        end
        push!(B_by_mid[mid], (src, B_block))
    end

    # Phase 1: Allocate result blocks
    # Identify all (tgt, src) pairs that will be non-zero
    for ((tgt, mid), _) in A.blocks
        if haskey(B_by_mid, mid)
            for (src, _) in B_by_mid[mid]
                key = (tgt, src)
                if !haskey(result_blocks, key)
                    # Allocate zero matrix of correct size
                    dim_tgt = size(A.blocks[(tgt, mid)], 1)
                    dim_src = size(B.blocks[(mid, src)], 2) # Access B via B.blocks to be safe/clean? No, B_block is available in loop if we restructure
                    # Wait, B_by_mid stores B_block.
                    # We can't easily access B dimensions in this loop without the block itself.
                    # Let's just allocate on first encounter in the main loop, but optimize the lookups.
                    
                    # Actually, the "Allocate on first encounter" pattern in the previous version IS the standard way.
                    # Separating allocation requires iterating twice.
                    # Let's stick to the single pass but optimize the inner loop slightly.
                end
            end
        end
    end
    
    # Let's retry the "Allocate First" strategy properly
    # We need the dimensions.
    
    # 1. Identify active keys and allocate
    for ((tgt, mid), A_block) in A.blocks
        if haskey(B_by_mid, mid)
            for (src, B_block) in B_by_mid[mid]
                key = (tgt, src)
                if !haskey(result_blocks, key)
                    # Allocate
                    result_blocks[key] = zeros(ComplexF64, size(A_block, 1), size(B_block, 2))
                end
            end
        end
    end

    # 2. Compute and accumulate
    # Now we know all keys exist, so we can skip haskey check? 
    # No, we still need to find the specific blocks to multiply.
    
    for ((tgt, mid), A_block) in A.blocks
        if haskey(B_by_mid, mid)
            for (src, B_block) in B_by_mid[mid]
                key = (tgt, src)
                # In-place accumulation
                # result += A * B
                mul!(result_blocks[key], A_block, B_block, 1.0, 1.0)
            end
        end
    end

    # Remove any blocks that became zero due to cancellation
    filter!(((key, block),) -> !iszero(block), result_blocks)

    return BlockDenseOperator(A.basis, result_flux, result_blocks)
end

"""
    Base.sum(ops::Vector{BlockDenseOperator}) -> BlockDenseOperator

Efficiently sum a vector of block-dense operators.

This optimization avoids intermediate allocations by determining the full structure
of the result first and then accumulating blocks in-place.

Arguments:
- `ops::Vector{BlockDenseOperator}`: List of operators to sum

Returns:
- `BlockDenseOperator`: Sum of all operators

Requirements:
- All operators must act on the same basis
- All operators must have the same flux
"""
function Base.sum(ops::Vector{BlockDenseOperator})
    isempty(ops) && error("Cannot sum empty list of operators")
    length(ops) == 1 && return ops[1]

    # Verify basis and flux consistency
    basis = ops[1].basis
    flux = ops[1].flux
    for i in 2:length(ops)
        @assert ops[i].basis === basis "Operators must act on same basis"
        @assert ops[i].flux.delta_qn == flux.delta_qn "Operators must have same flux to be added"
    end

    # Collect all unique block keys (target, source) present in any operator
    # and pre-calculate the final block structure
    # This avoids resizing dictionaries repeatedly
    all_keys = Set{Tuple{Int,Int}}()
    for op in ops
        for key in keys(op.blocks)
            push!(all_keys, key)
        end
    end

    # Initialize result blocks
    result_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    
    # Iterate over all possible keys
    for key in all_keys
        (tgt, src) = key
        dim_tgt = basis.sectors[tgt].dim
        dim_src = basis.sectors[src].dim
        
        # Initialize accumulator for this block
        # We start with zero matrix
        acc = zeros(ComplexF64, dim_tgt, dim_src)
        
        # Accumulate contributions from all operators that have this block
        is_nonzero = false
        for op in ops
            if haskey(op.blocks, key)
                # acc .+= op.blocks[key]
                # Use broadcast for in-place addition
                acc .+= op.blocks[key]
                is_nonzero = true
            end
        end
        
        # Store if non-zero (or even if zero, but typically we store non-zeros)
        # Here we only store if at least one operator had a block there. 
        # Ideally we should check if `acc` itself is zero, but standard sparse logic
        # often keeps structural non-zeros. Let's check numerical zero to be cleaner.
        if is_nonzero && !iszero(acc)
            result_blocks[key] = acc
        end
    end

    return BlockDenseOperator(basis, flux, result_blocks)
end

"""
    Base.:+(A::BlockDenseOperator, B::BlockDenseOperator) -> BlockDenseOperator

Add two block-dense operators with the same flux.

Arguments:
- `A::BlockDenseOperator`: First operator
- `B::BlockDenseOperator`: Second operator

Returns:
- `BlockDenseOperator`: Sum operator A+B

Requirements:
- Both operators must have the same flux
- Both operators must act on the same basis

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blockdense(basis, 1)
c2 = get_annihilator_blockdense(basis, 2)

# Both have flux QN_N(-1), so can be added
c_sum = c1 + c2
```
"""
function Base.:+(A::BlockDenseOperator, B::BlockDenseOperator)
    @assert A.basis === B.basis "Operators must act on same basis"
    @assert A.flux.delta_qn == B.flux.delta_qn "Operators must have same flux to be added"

    # Start with copy of A's blocks
    result_blocks = copy(A.blocks)

    # Add B's blocks
    for (key, B_block) in B.blocks
        if haskey(result_blocks, key)
            result_blocks[key] += B_block
        else
            result_blocks[key] = copy(B_block)
        end
    end

    # Remove any blocks that became zero
    filter!(((key, block),) -> !iszero(block), result_blocks)

    return BlockDenseOperator(A.basis, A.flux, result_blocks)
end

"""
    Base.:-(A::BlockDenseOperator, B::BlockDenseOperator) -> BlockDenseOperator

Subtract two block-dense operators with the same flux.

Equivalent to A + (-1)*B.
"""
function Base.:-(A::BlockDenseOperator, B::BlockDenseOperator)
    return A + (-1) * B
end

"""
    Base.:*(α::Number, A::BlockDenseOperator) -> BlockDenseOperator

Multiply operator by scalar.

Arguments:
- `α::Number`: Scalar coefficient
- `A::BlockDenseOperator`: Operator

Returns:
- `BlockDenseOperator`: Scaled operator α*A with same flux

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blockdense(basis, 1)

# Scale by coefficient
c1_scaled = 2.0 * c1
```
"""
function Base.:*(α::Number, A::BlockDenseOperator)
    if iszero(α)
        # Return zero operator (no blocks)
        return BlockDenseOperator(A.basis, A.flux, Dict{Tuple{Int,Int}, Matrix{ComplexF64}}())
    end

    # Scale all blocks
    scaled_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    for (key, block) in A.blocks
        scaled_blocks[key] = α * block
    end

    return BlockDenseOperator(A.basis, A.flux, scaled_blocks)
end

"""
    Base.:*(A::BlockDenseOperator, α::Number) -> BlockDenseOperator

Multiply operator by scalar (commutative version).
"""
Base.:*(A::BlockDenseOperator, α::Number) = α * A

"""
    Base.:/(A::BlockDenseOperator, α::Number) -> BlockDenseOperator

Divide operator by scalar.

Equivalent to (1/α) * A.
"""
Base.:/(A::BlockDenseOperator, α::Number) = (1/α) * A

"""
    Base.adjoint(A::BlockDenseOperator) -> BlockDenseOperator

Compute the adjoint (Hermitian conjugate) of a block-dense operator.

The adjoint operator has:
- Reversed flux: flux(A') = reverse_qn(flux(A))
- Transposed and conjugated blocks: A'[src, tgt] = conj(A[tgt, src]ᵀ)

Arguments:
- `A::BlockDenseOperator`: Input operator

Returns:
- `BlockDenseOperator`: Adjoint operator A'

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blockdense(basis, 1)

# Creation operator
c1_dag = c1'
# flux(c1_dag) = reverse_qn(QN_N(-1)) = QN_N(1)
```
"""
function Base.adjoint(A::BlockDenseOperator)
    # Reverse flux
    adj_flux = OperatorFlux(reverse_qn(A.flux.delta_qn))

    # Transpose and conjugate blocks, swapping indices
    adj_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    for ((tgt, src), block) in A.blocks
        # A'[src, tgt] = (A[tgt, src])'
        # Note: Julia's adjoint (') does conjugate transpose
        adj_blocks[(src, tgt)] = adjoint(block)
    end

    return BlockDenseOperator(A.basis, adj_flux, adj_blocks)
end

"""
    Base.:-(A::BlockDenseOperator) -> BlockDenseOperator

Unary negation of operator.

Equivalent to (-1) * A.
"""
Base.:-(A::BlockDenseOperator) = (-1) * A
