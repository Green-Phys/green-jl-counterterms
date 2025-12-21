"""
Dense operators for quantum many-body systems.

Operators are stored as dense matrices, wrapped in a struct that carries
basis information for API consistency with BlockDenseOperator.
"""

"""
    DenseOperator

A dense matrix operator with associated basis.

Fields:
- `basis::DenseBasis`: The basis this operator acts on
- `data::Matrix{ComplexF64}`: The operator matrix

This wrapper provides API symmetry with `BlockDenseOperator` while
maintaining full access to the underlying matrix via the `data` field.

Example:
```julia
basis = make_basis_dense(2)
c = get_annihilators_dense(basis)

# Access underlying matrix
matrix = c[1].data

# Operator algebra works naturally
n1 = c[1]' * c[1]  # Number operator
```
"""
struct DenseOperator
    basis::DenseBasis
    data::Matrix{ComplexF64}

    function DenseOperator(basis::DenseBasis, data::Matrix{ComplexF64})
        @assert size(data) == (basis.dim, basis.dim) "Matrix size must match basis dimension"
        new(basis, data)
    end
end

# Forward size to the underlying matrix
Base.size(op::DenseOperator) = size(op.data)
Base.size(op::DenseOperator, dim::Int) = size(op.data, dim)

# Forward ishermitian to the underlying matrix
LinearAlgebra.ishermitian(op::DenseOperator) = ishermitian(op.data)

# Display
function Base.show(io::IO, op::DenseOperator)
    println(io, "DenseOperator")
    println(io, "  basis: ", op.basis)
    println(io, "  matrix: $(size(op.data, 1))×$(size(op.data, 2)) ComplexF64")
end
