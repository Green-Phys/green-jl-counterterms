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
