"""
Hamiltonian diagonalization for dense method.

Solves the eigenvalue problem for dense Hamiltonians.
"""

"""
    DenseEigenResult

Result of dense eigenvalue decomposition.

Fields:
- `eigenvalues::DenseDiagonalOperator`: Eigenvalues as a diagonal operator
- `eigenvectors::DenseOperator`: Eigenvectors as a unitary operator

The eigenvectors matrix U transforms from the computational basis to the
eigenbasis: if |ψ_i⟩ is the i-th eigenvector, then U[:, i] = |ψ_i⟩.

Example:
```julia
H, c = build_hamiltonian_dense("hopping.txt", "coulomb.txt")
result = solve_hamiltonian_dense(H)

# Access eigenvalues
evals = result.eigenvalues.diag  # Vector{Float64}

# Access eigenvectors
evecs = result.eigenvectors.data  # Matrix{ComplexF64}

# Transform operator to eigenbasis
A_eigenbasis = evecs' * A.data * evecs
```
"""
struct DenseEigenResult
    eigenvalues::DenseDiagonalOperator
    eigenvectors::DenseOperator
end

"""
    solve_hamiltonian_dense(H::DenseOperator) -> DenseEigenResult

Diagonalize a Hamiltonian and return eigenvalues and eigenvectors.

Arguments:
- `H::DenseOperator`: Hamiltonian operator

Returns:
- `DenseEigenResult`: Result object containing eigenvalues and eigenvectors
  as operator types.

Example:
```julia
H, c = build_hamiltonian_dense("hopping.txt", "coulomb.txt")
result = solve_hamiltonian_dense(H)

# Access eigenvalues
E = result.eigenvalues.diag

# Access eigenvectors
U = result.eigenvectors.data
```

Implementation:
Uses standard dense eigenvalue decomposition via Hermitian solver.
"""
function solve_hamiltonian_dense(H::DenseOperator)
    # Diagonalize using Hermitian solver for efficiency
    sol = eigen(Hermitian(H.data))

    # Wrap results in operator types
    eigenvalues = DenseDiagonalOperator(H.basis, sol.values)
    eigenvectors = DenseOperator(H.basis, sol.vectors)

    return DenseEigenResult(eigenvalues, eigenvectors)
end
