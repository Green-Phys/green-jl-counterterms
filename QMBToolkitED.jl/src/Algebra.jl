"""
Linear algebra operations for operators.

Implements:
- Operator multiplication: A * B (block chaining with flux composition)
- Operator addition: A + B (flux must match)
- Scalar multiplication: α * A
- Adjoint: A' (flux reversal and block transposition)
"""

"""
    Base.:*(A::Operator, B::Operator) -> Operator

Multiply two operators by chaining blocks.

The resulting operator has flux = A.flux + B.flux.
Block (target, source) in the result is computed by summing over intermediate sectors:
    result[target, source] = Σ_mid A.blocks[target, mid] * B.blocks[mid, source]

Arguments:
- `A::Operator`: Left operator
- `B::Operator`: Right operator

Returns:
- `Operator`: Product operator A*B

Requirements:
- Both operators must act on the same basis
- For efficiency, only non-zero blocks are computed

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
c1 = annihilator(basis, 1)
c1_dag = c1'

# Number operator: n1 = c1† * c1
n1 = c1_dag * c1
# flux(n1) = QN_N(1) + QN_N(-1) = QN_N(0)
```
"""
function Base.:*(A::Operator, B::Operator)
    @assert A.basis === B.basis "Operators must act on same basis"

    # Result flux is composition of input fluxes
    result_flux = A.flux + B.flux

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

    # Allocate on first encounter, then accumulate in-place
    for ((tgt, mid), A_block) in A.blocks
        if haskey(B_by_mid, mid)
            for (src, B_block) in B_by_mid[mid]
                key = (tgt, src)
                if !haskey(result_blocks, key)
                    # First contribution: allocate uninitialized and overwrite
                    C = Matrix{ComplexF64}(undef, size(A_block, 1), size(B_block, 2))
                    mul!(C, A_block, B_block)
                    result_blocks[key] = C
                else
                    # Subsequent contributions: accumulate
                    mul!(result_blocks[key], A_block, B_block, 1.0, 1.0)
                end
            end
        end
    end

    # Remove any blocks that became zero due to cancellation
    filter!(((key, block),) -> !iszero(block), result_blocks)

    return Operator(A.basis, result_flux, result_blocks)
end

"""
    Base.sum(ops::Vector{Operator}) -> Operator

Efficiently sum a vector of operators.

This optimization avoids intermediate allocations by determining the full structure
of the result first and then accumulating blocks in-place.

Arguments:
- `ops::Vector{Operator}`: List of operators to sum

Returns:
- `Operator`: Sum of all operators

Requirements:
- All operators must act on the same basis
- All operators must have the same flux
"""
function Base.sum(ops::Vector{Operator})
    isempty(ops) && error("Cannot sum empty list of operators")
    length(ops) == 1 && return ops[1]

    # Verify basis and flux consistency
    basis = ops[1].basis
    flux = ops[1].flux
    for i in 2:length(ops)
        @assert ops[i].basis === basis "Operators must act on same basis"
        @assert ops[i].flux == flux "Operators must have same flux to be added"
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
        # Find the first operator that has this block
        first_idx = 0
        for i in 1:length(ops)
            if haskey(ops[i].blocks, key)
                first_idx = i
                break
            end
        end

        # This should always be found since key came from all_keys
        if first_idx > 0
            # Initialize with a copy of the first contributing block
            acc = copy(ops[first_idx].blocks[key])

            # Accumulate contributions from remaining operators
            for i in (first_idx + 1):length(ops)
                if haskey(ops[i].blocks, key)
                    acc .+= ops[i].blocks[key]
                end
            end

            # Store if non-zero
            if !iszero(acc)
                result_blocks[key] = acc
            end
        end
    end

    return Operator(basis, flux, result_blocks)
end

"""
    Base.:+(A::Operator, B::Operator) -> Operator

Add two operators with the same flux.

Arguments:
- `A::Operator`: First operator
- `B::Operator`: Second operator

Returns:
- `Operator`: Sum operator A+B

Requirements:
- Both operators must have the same flux
- Both operators must act on the same basis

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
c1 = annihilator(basis, 1)
c2 = annihilator(basis, 2)

# Both have flux QN_N(-1), so can be added
c_sum = c1 + c2
```
"""
function Base.:+(A::Operator, B::Operator)
    @assert A.basis === B.basis "Operators must act on same basis"
    @assert A.flux == B.flux "Operators must have same flux to be added"

    result_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    # Blocks in A: copy or add with B
    for (key, A_block) in A.blocks
        if haskey(B.blocks, key)
            result_blocks[key] = A_block + B.blocks[key]
        else
            result_blocks[key] = copy(A_block)
        end
    end

    # Blocks only in B
    for (key, B_block) in B.blocks
        if !haskey(A.blocks, key)
            result_blocks[key] = copy(B_block)
        end
    end

    # Remove any blocks that became zero
    filter!(((key, block),) -> !iszero(block), result_blocks)

    return Operator(A.basis, A.flux, result_blocks)
end

"""
    Base.:-(A::Operator, B::Operator) -> Operator

Subtract two operators with the same flux.

Equivalent to A + (-1)*B.
"""
function Base.:-(A::Operator, B::Operator)
    return A + (-1) * B
end

"""
    Base.:*(α::Number, A::Operator) -> Operator

Multiply operator by scalar.

Arguments:
- `α::Number`: Scalar coefficient
- `A::Operator`: Operator

Returns:
- `Operator`: Scaled operator α*A with same flux

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
c1 = annihilator(basis, 1)

# Scale by coefficient
c1_scaled = 2.0 * c1
```
"""
function Base.:*(α::Number, A::Operator)
    if iszero(α)
        # Return zero operator (no blocks)
        return Operator(A.basis, A.flux, Dict{Tuple{Int,Int}, Matrix{ComplexF64}}())
    end

    # Scale all blocks
    scaled_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    for (key, block) in A.blocks
        scaled_blocks[key] = α * block
    end

    return Operator(A.basis, A.flux, scaled_blocks)
end

"""
    Base.:*(A::Operator, α::Number) -> Operator

Multiply operator by scalar (commutative version).
"""
Base.:*(A::Operator, α::Number) = α * A

"""
    Base.:/(A::Operator, α::Number) -> Operator

Divide operator by scalar.

Equivalent to (1/α) * A.
"""
Base.:/(A::Operator, α::Number) = (1/α) * A

"""
    Base.adjoint(A::Operator) -> Operator

Compute the adjoint (Hermitian conjugate) of an operator.

The adjoint operator has:
- Reversed flux: flux(A') = -flux(A)
- Transposed and conjugated blocks: A'[src, tgt] = conj(A[tgt, src]ᵀ)

Arguments:
- `A::Operator`: Input operator

Returns:
- `Operator`: Adjoint operator A'

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
c1 = annihilator(basis, 1)

# Creation operator
c1_dag = c1'
# flux(c1_dag) = -QN_N(-1) = QN_N(1)
```
"""
function Base.adjoint(A::Operator)
    # Reverse flux
    adj_flux = -A.flux

    # Transpose and conjugate blocks, swapping indices
    adj_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    for ((tgt, src), block) in A.blocks
        # A'[src, tgt] = (A[tgt, src])'
        # Note: Julia's adjoint (') does conjugate transpose
        adj_blocks[(src, tgt)] = adjoint(block)
    end

    return Operator(A.basis, adj_flux, adj_blocks)
end

"""
    Base.:-(A::Operator) -> Operator

Unary negation of operator.

Equivalent to (-1) * A.
"""
Base.:-(A::Operator) = (-1) * A
