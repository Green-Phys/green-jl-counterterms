"""
Hamiltonian diagonalization for dense method.

Solves the eigenvalue problem for dense Hamiltonians.
"""

"""
    solve_hamiltonian_dense(H) -> (Vector{Float64}, Matrix{ComplexF64})

Diagonalize a Hamiltonian matrix and return eigenvalues and eigenvectors.

Arguments:
- `H::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}}`: Hamiltonian matrix

Returns:
- `Vector{Float64}`: Eigenvalues (sorted in ascending order)
- `Matrix{ComplexF64}`: Eigenvectors (columns are eigenvectors)

Example:
```julia
H, c = build_hamiltonian_dense("hopping.txt", "coulomb.txt")
evals, evecs = solve_hamiltonian_dense(H)
```

Implementation:
Uses standard dense eigenvalue decomposition. If input is sparse, converts to dense first.
"""
function solve_hamiltonian_dense(H::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64}})
    # Convert to dense if needed
    H_dense = H isa Matrix ? H : Matrix(H)

    # Diagonalize using Hermitian solver for efficiency
    sol = eigen(Hermitian(H_dense))

    return sol.values, sol.vectors
end
