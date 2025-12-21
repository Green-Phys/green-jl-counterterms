"""
Hamiltonian diagonalization for dense method.

Solves the eigenvalue problem for dense Hamiltonians.
"""

"""
    DenseEigenResult

Result of dense eigenvalue decomposition.

Fields:
- `evals::Vector{Float64}`: Eigenvalues
- `evecs::Matrix{ComplexF64}`: Eigenvectors
- `basis::DenseBasis`: The basis used

This object is iterable, allowing destructuring: `evals, evecs = result`
"""
struct DenseEigenResult
    evals::Vector{Float64}
    evecs::Matrix{ComplexF64}
    basis::DenseBasis
end

"""
    get_eigenvalues(result::DenseEigenResult) -> Vector{Float64}

Get eigenvalues from DenseEigenResult.
"""
get_eigenvalues(result::DenseEigenResult) = result.evals

"""
    get_eigenvectors(result::DenseEigenResult) -> Matrix{ComplexF64}

Get eigenvectors from DenseEigenResult.
"""
get_eigenvectors(result::DenseEigenResult) = result.evecs

"""
    solve_hamiltonian_dense(H::DenseOperator) -> DenseEigenResult

Diagonalize a Hamiltonian and return eigenvalues and eigenvectors.

Arguments:
- `H::DenseOperator`: Hamiltonian operator

Returns:
- `DenseEigenResult`: Result object containing eigenvalues, eigenvectors, and basis.
  This object is iterable for backward compatibility: `evals, evecs = result`

Example:
```julia
H, c = build_hamiltonian_dense("hopping.txt", "coulomb.txt")
result = solve_hamiltonian_dense(H)
evals, evecs = result  # Tuple unpacking works
```

Implementation:
Uses standard dense eigenvalue decomposition via Hermitian solver.
"""
function solve_hamiltonian_dense(H::DenseOperator)
    # Diagonalize using Hermitian solver for efficiency
    sol = eigen(Hermitian(H.data))

    return DenseEigenResult(sol.values, sol.vectors, H.basis)
end

