"""
Diagonal operators for dense Hilbert space representation.

Diagonal operators store only the diagonal elements, providing efficient
storage for eigenvalues and other diagonal matrices.
"""

"""
    DenseDiagonalOperator

A diagonal operator in the full Hilbert space, storing only diagonal elements.

Fields:
- `basis::DenseBasis`: The Hilbert space basis
- `diag::Vector{Float64}`: Diagonal elements

This type is used to efficiently store eigenvalues from Hamiltonian diagonalization.
"""
struct DenseDiagonalOperator
    basis::DenseBasis
    diag::Vector{Float64}

    function DenseDiagonalOperator(basis::DenseBasis, diag::Vector{Float64})
        @assert length(diag) == basis.dim "Diagonal length ($(length(diag))) must match basis dimension ($(basis.dim))"
        new(basis, diag)
    end
end

# Allow construction from any real vector
DenseDiagonalOperator(basis::DenseBasis, diag::Vector{<:Real}) =
    DenseDiagonalOperator(basis, convert(Vector{Float64}, diag))

"""
    Base.size(D::DenseDiagonalOperator)

Return the size of the diagonal operator as a matrix.
"""
Base.size(D::DenseDiagonalOperator) = (D.basis.dim, D.basis.dim)

"""
    Base.getindex(D::DenseDiagonalOperator, i::Int)

Get the i-th diagonal element.
"""
Base.getindex(D::DenseDiagonalOperator, i::Int) = D.diag[i]

"""
    Base.length(D::DenseDiagonalOperator)

Return the number of diagonal elements.
"""
Base.length(D::DenseDiagonalOperator) = length(D.diag)
