"""
Conversion between DenseOperator and Matrix{ComplexF64}.

Functions:
- to_matrix: Convert DenseOperator → Matrix{ComplexF64}
"""

"""
    to_matrix(op::DenseOperator) -> Matrix{ComplexF64}

Extract the underlying matrix from a DenseOperator.

This is the identity operation for dense operators, provided for
API symmetry with BlockDenseOperator.

Arguments:
- `op::DenseOperator`: Dense operator

Returns:
- `Matrix{ComplexF64}`: The underlying matrix

Example:
```julia
basis = make_basis_dense(2)
c = get_annihilators_dense(basis)

# Convert to matrix
c1_mat = to_matrix(c[1])
@assert size(c1_mat) == (4, 4)  # 2^2 = 4
```
"""
to_matrix(op::DenseOperator) = op.data

"""
    to_matrix(D::DenseDiagonalOperator) -> Matrix{ComplexF64}

Convert a diagonal operator to a full matrix.

Arguments:
- `D::DenseDiagonalOperator`: Diagonal operator

Returns:
- `Matrix{ComplexF64}`: Full diagonal matrix

Example:
```julia
basis = make_basis_dense(2)
D = DenseDiagonalOperator(basis, [1.0, 2.0, 3.0, 4.0])
mat = to_matrix(D)
@assert mat[1,1] == 1.0 && mat[2,2] == 2.0
```
"""
function to_matrix(D::DenseDiagonalOperator)
    return Matrix{ComplexF64}(Diagonal(D.diag))
end

"""
    to_vector(D::DenseDiagonalOperator) -> Vector{Float64}

Extract the diagonal values as a vector.

Arguments:
- `D::DenseDiagonalOperator`: Diagonal operator

Returns:
- `Vector{Float64}`: The diagonal elements
"""
to_vector(D::DenseDiagonalOperator) = D.diag
