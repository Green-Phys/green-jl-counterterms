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

    # For each potential (target, source) pair in result:
    # result[target, source] = Σ_mid A[target, mid] * B[mid, source]

    # Strategy: iterate over A's blocks and B's blocks
    # If A connects (tgt, mid) and B connects (mid, src), contribute to result[tgt, src]

    for ((a_tgt, a_mid), A_block) in A.blocks
        for ((b_mid, b_src), B_block) in B.blocks
            # Check if intermediate sectors match
            if a_mid == b_mid
                # This pair contributes to result[a_tgt, b_src]
                key = (a_tgt, b_src)

                # Compute contribution: A_block * B_block
                contrib = A_block * B_block

                # Add to result (accumulate if key already exists)
                if haskey(result_blocks, key)
                    result_blocks[key] += contrib
                else
                    result_blocks[key] = contrib
                end
            end
        end
    end

    # Remove any blocks that became zero due to cancellation
    filter!(((key, block),) -> !iszero(block), result_blocks)

    return BlockDenseOperator(A.basis, result_flux, result_blocks)
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
