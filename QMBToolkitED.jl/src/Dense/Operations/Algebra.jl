"""
Linear algebra operations for dense operators.

Implements:
- Operator multiplication: A * B
- Operator addition: A + B
- Scalar multiplication: α * A
- Adjoint: A'
"""

"""
    Base.:*(A::DenseOperator, B::DenseOperator) -> DenseOperator

Multiply two dense operators.

Arguments:
- `A::DenseOperator`: Left operator
- `B::DenseOperator`: Right operator

Returns:
- `DenseOperator`: Product operator A*B

Requirements:
- Both operators must act on the same basis

Example:
```julia
basis = make_basis_dense(2)
c = get_annihilators_dense(basis)

# Number operator: n1 = c1† * c1
n1 = c[1]' * c[1]
```
"""
function Base.:*(A::DenseOperator, B::DenseOperator)
    @assert A.basis === B.basis "Operators must act on same basis"
    return DenseOperator(A.basis, A.data * B.data)
end

"""
    Base.sum(ops::Vector{DenseOperator}) -> DenseOperator

Efficiently sum a vector of dense operators.

Arguments:
- `ops::Vector{DenseOperator}`: List of operators to sum

Returns:
- `DenseOperator`: Sum of all operators

Requirements:
- All operators must act on the same basis
"""
function Base.sum(ops::Vector{DenseOperator})
    isempty(ops) && error("Cannot sum empty list of operators")
    length(ops) == 1 && return ops[1]

    # Verify basis consistency
    basis = ops[1].basis
    for i in 2:length(ops)
        @assert ops[i].basis === basis "Operators must act on same basis"
    end

    # Sum all matrices
    result = copy(ops[1].data)
    for i in 2:length(ops)
        result .+= ops[i].data
    end

    return DenseOperator(basis, result)
end

"""
    Base.:+(A::DenseOperator, B::DenseOperator) -> DenseOperator

Add two dense operators.

Arguments:
- `A::DenseOperator`: First operator
- `B::DenseOperator`: Second operator

Returns:
- `DenseOperator`: Sum operator A+B

Requirements:
- Both operators must act on the same basis

Example:
```julia
basis = make_basis_dense(2)
c = get_annihilators_dense(basis)

# Sum of annihilation operators
c_sum = c[1] + c[2]
```
"""
function Base.:+(A::DenseOperator, B::DenseOperator)
    @assert A.basis === B.basis "Operators must act on same basis"
    return DenseOperator(A.basis, A.data + B.data)
end

"""
    Base.:-(A::DenseOperator, B::DenseOperator) -> DenseOperator

Subtract two dense operators.

Equivalent to A + (-1)*B.
"""
function Base.:-(A::DenseOperator, B::DenseOperator)
    @assert A.basis === B.basis "Operators must act on same basis"
    return DenseOperator(A.basis, A.data - B.data)
end

"""
    Base.:*(α::Number, A::DenseOperator) -> DenseOperator

Multiply operator by scalar.

Arguments:
- `α::Number`: Scalar coefficient
- `A::DenseOperator`: Operator

Returns:
- `DenseOperator`: Scaled operator α*A

Example:
```julia
basis = make_basis_dense(2)
c = get_annihilators_dense(basis)

# Scale by coefficient
c1_scaled = 2.0 * c[1]
```
"""
function Base.:*(α::Number, A::DenseOperator)
    return DenseOperator(A.basis, α * A.data)
end

"""
    Base.:*(A::DenseOperator, α::Number) -> DenseOperator

Multiply operator by scalar (commutative version).
"""
Base.:*(A::DenseOperator, α::Number) = α * A

"""
    Base.:/(A::DenseOperator, α::Number) -> DenseOperator

Divide operator by scalar.

Equivalent to (1/α) * A.
"""
Base.:/(A::DenseOperator, α::Number) = (1/α) * A

"""
    Base.adjoint(A::DenseOperator) -> DenseOperator

Compute the adjoint (Hermitian conjugate) of a dense operator.

Arguments:
- `A::DenseOperator`: Input operator

Returns:
- `DenseOperator`: Adjoint operator A'

Example:
```julia
basis = make_basis_dense(2)
c = get_annihilators_dense(basis)

# Creation operator
c1_dag = c[1]'
```
"""
function Base.adjoint(A::DenseOperator)
    # Collect to Matrix to avoid Adjoint wrapper type
    return DenseOperator(A.basis, collect(adjoint(A.data)))
end

"""
    Base.:-(A::DenseOperator) -> DenseOperator

Unary negation of operator.

Equivalent to (-1) * A.
"""
Base.:-(A::DenseOperator) = (-1) * A

# ============================================================================
# DenseDiagonalOperator Operations
# ============================================================================

"""
    Base.:*(D::DenseDiagonalOperator, A::DenseOperator) -> DenseOperator

Multiply diagonal operator by dense operator (column scaling).

D * A scales each column j of A by D[j].
"""
function Base.:*(D::DenseDiagonalOperator, A::DenseOperator)
    @assert D.basis === A.basis "Operators must act on same basis"
    # D * A: scales columns of A
    return DenseOperator(A.basis, Diagonal(D.diag) * A.data)
end

"""
    Base.:*(A::DenseOperator, D::DenseDiagonalOperator) -> DenseOperator

Multiply dense operator by diagonal operator (row scaling).

A * D scales each row i of A by D[i].
"""
function Base.:*(A::DenseOperator, D::DenseDiagonalOperator)
    @assert A.basis === D.basis "Operators must act on same basis"
    # A * D: scales rows of A
    return DenseOperator(A.basis, A.data * Diagonal(D.diag))
end

"""
    Base.:*(D1::DenseDiagonalOperator, D2::DenseDiagonalOperator) -> DenseDiagonalOperator

Multiply two diagonal operators (element-wise multiplication).
"""
function Base.:*(D1::DenseDiagonalOperator, D2::DenseDiagonalOperator)
    @assert D1.basis === D2.basis "Operators must act on same basis"
    return DenseDiagonalOperator(D1.basis, D1.diag .* D2.diag)
end

"""
    Base.:+(D1::DenseDiagonalOperator, D2::DenseDiagonalOperator) -> DenseDiagonalOperator

Add two diagonal operators.
"""
function Base.:+(D1::DenseDiagonalOperator, D2::DenseDiagonalOperator)
    @assert D1.basis === D2.basis "Operators must act on same basis"
    return DenseDiagonalOperator(D1.basis, D1.diag .+ D2.diag)
end

"""
    Base.:-(D1::DenseDiagonalOperator, D2::DenseDiagonalOperator) -> DenseDiagonalOperator

Subtract two diagonal operators.
"""
function Base.:-(D1::DenseDiagonalOperator, D2::DenseDiagonalOperator)
    @assert D1.basis === D2.basis "Operators must act on same basis"
    return DenseDiagonalOperator(D1.basis, D1.diag .- D2.diag)
end

"""
    Base.:*(α::Number, D::DenseDiagonalOperator) -> DenseDiagonalOperator

Multiply diagonal operator by scalar.
"""
function Base.:*(α::Number, D::DenseDiagonalOperator)
    return DenseDiagonalOperator(D.basis, α .* D.diag)
end

"""
    Base.:*(D::DenseDiagonalOperator, α::Number) -> DenseDiagonalOperator

Multiply diagonal operator by scalar (commutative version).
"""
Base.:*(D::DenseDiagonalOperator, α::Number) = α * D

"""
    Base.:/(D::DenseDiagonalOperator, α::Number) -> DenseDiagonalOperator

Divide diagonal operator by scalar.
"""
Base.:/(D::DenseDiagonalOperator, α::Number) = (1/α) * D

"""
    Base.adjoint(D::DenseDiagonalOperator) -> DenseDiagonalOperator

Compute the adjoint of a diagonal operator.

For real diagonal operators, the adjoint is the same as the original.
"""
function Base.adjoint(D::DenseDiagonalOperator)
    return D  # Real diagonal is self-adjoint
end

"""
    Base.:-(D::DenseDiagonalOperator) -> DenseDiagonalOperator

Unary negation of diagonal operator.
"""
Base.:-(D::DenseDiagonalOperator) = (-1) * D
