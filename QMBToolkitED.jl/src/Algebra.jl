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

The resulting operator has flux = combine_qn(A.flux, B.flux).
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
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator(basis, 1)
c1_dag = c1'

# Number operator: n1 = c1† * c1
n1 = c1_dag * c1
# flux(n1) = QN_N(1) + QN_N(-1) = QN_N(0)
```
"""
function Base.:*(A::Operator, B::Operator)
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

        # Store if non-zero
        if is_nonzero && !iszero(acc)
            result_blocks[key] = acc
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
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator(basis, 1)
c2 = get_annihilator(basis, 2)

# Both have flux QN_N(-1), so can be added
c_sum = c1 + c2
```
"""
function Base.:+(A::Operator, B::Operator)
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
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator(basis, 1)

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
- Reversed flux: flux(A') = reverse_qn(flux(A))
- Transposed and conjugated blocks: A'[src, tgt] = conj(A[tgt, src]ᵀ)

Arguments:
- `A::Operator`: Input operator

Returns:
- `Operator`: Adjoint operator A'

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator(basis, 1)

# Creation operator
c1_dag = c1'
# flux(c1_dag) = reverse_qn(QN_N(-1)) = QN_N(1)
```
"""
function Base.adjoint(A::Operator)
    # Reverse flux
    adj_flux = OperatorFlux(reverse_qn(A.flux.delta_qn))

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

# ============================================================================
# DiagonalOperator Operations
# ============================================================================

"""
    Base.:*(D::DiagonalOperator, A::Operator) -> Operator

Multiply diagonal operator by operator.

For each block (tgt, src), scales rows by D's diagonal values for sector tgt.
"""
function Base.:*(D::DiagonalOperator, A::Operator)
    @assert D.basis === A.basis "Operators must act on same basis"

    # Result has same flux as A (D has zero flux)
    result_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    for ((tgt, src), block) in A.blocks
        # D * A: left-multiply by diagonal scales rows
        # (D * A)[tgt, src] = D[tgt, tgt] * A[tgt, src]
        if haskey(D.sector_diags, tgt)
            diag_tgt = D.sector_diags[tgt]
            result_blocks[(tgt, src)] = Diagonal(diag_tgt) * block
        end
    end

    # Remove zero blocks
    filter!(((key, block),) -> !iszero(block), result_blocks)

    return Operator(A.basis, A.flux, result_blocks)
end

"""
    Base.:*(A::Operator, D::DiagonalOperator) -> Operator

Multiply operator by diagonal operator.

For each block (tgt, src), scales columns by D's diagonal values for sector src.
"""
function Base.:*(A::Operator, D::DiagonalOperator)
    @assert A.basis === D.basis "Operators must act on same basis"

    # Result has same flux as A (D has zero flux)
    result_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    for ((tgt, src), block) in A.blocks
        if haskey(D.sector_diags, src)
            # A * D: right-multiply by diagonal scales columns
            # (A * D)[tgt, src] = A[tgt, src] * D[src, src]
            diag_src = D.sector_diags[src]
            result_blocks[(tgt, src)] = block * Diagonal(diag_src)
        end
    end

    # Remove zero blocks
    filter!(((key, block),) -> !iszero(block), result_blocks)

    return Operator(A.basis, A.flux, result_blocks)
end

"""
    Base.:*(D1::DiagonalOperator, D2::DiagonalOperator) -> DiagonalOperator

Multiply two diagonal operators (element-wise per sector).
"""
function Base.:*(D1::DiagonalOperator, D2::DiagonalOperator)
    @assert D1.basis === D2.basis "Operators must act on same basis"

    result_diags = Dict{Int, Vector{Float64}}()

    # Only sectors present in both
    for sector_idx in keys(D1.sector_diags)
        if haskey(D2.sector_diags, sector_idx)
            result_diags[sector_idx] = D1.sector_diags[sector_idx] .* D2.sector_diags[sector_idx]
        end
    end

    return DiagonalOperator(D1.basis, result_diags)
end

"""
    Base.:+(D1::DiagonalOperator, D2::DiagonalOperator) -> DiagonalOperator

Add two diagonal operators.
"""
function Base.:+(D1::DiagonalOperator, D2::DiagonalOperator)
    @assert D1.basis === D2.basis "Operators must act on same basis"

    result_diags = Dict{Int, Vector{Float64}}()

    # Union of sectors
    all_sectors = union(keys(D1.sector_diags), keys(D2.sector_diags))

    for sector_idx in all_sectors
        if haskey(D1.sector_diags, sector_idx) && haskey(D2.sector_diags, sector_idx)
            result_diags[sector_idx] = D1.sector_diags[sector_idx] .+ D2.sector_diags[sector_idx]
        elseif haskey(D1.sector_diags, sector_idx)
            result_diags[sector_idx] = copy(D1.sector_diags[sector_idx])
        else
            result_diags[sector_idx] = copy(D2.sector_diags[sector_idx])
        end
    end

    return DiagonalOperator(D1.basis, result_diags)
end

"""
    Base.:-(D1::DiagonalOperator, D2::DiagonalOperator) -> DiagonalOperator

Subtract two diagonal operators.
"""
function Base.:-(D1::DiagonalOperator, D2::DiagonalOperator)
    return D1 + (-1) * D2
end

"""
    Base.:*(α::Number, D::DiagonalOperator) -> DiagonalOperator

Multiply diagonal operator by scalar.
"""
function Base.:*(α::Number, D::DiagonalOperator)
    result_diags = Dict{Int, Vector{Float64}}(
        sector_idx => α .* diag for (sector_idx, diag) in D.sector_diags
    )
    return DiagonalOperator(D.basis, result_diags)
end

"""
    Base.:*(D::DiagonalOperator, α::Number) -> DiagonalOperator

Multiply diagonal operator by scalar (commutative version).
"""
Base.:*(D::DiagonalOperator, α::Number) = α * D

"""
    Base.:/(D::DiagonalOperator, α::Number) -> DiagonalOperator

Divide diagonal operator by scalar.
"""
Base.:/(D::DiagonalOperator, α::Number) = (1/α) * D

"""
    Base.adjoint(D::DiagonalOperator) -> DiagonalOperator

Compute the adjoint of a diagonal operator.

For real diagonal operators, the adjoint is the same as the original.
The flux remains zero.
"""
function Base.adjoint(D::DiagonalOperator)
    return D  # Real diagonal is self-adjoint
end

"""
    Base.:-(D::DiagonalOperator) -> DiagonalOperator

Unary negation of diagonal operator.
"""
Base.:-(D::DiagonalOperator) = (-1) * D
