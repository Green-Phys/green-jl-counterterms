"""
Dense module for quantum many-body exact diagonalization using full Hilbert space.

This module implements the traditional full Hilbert space approach, where all 2^n
basis states are explicitly represented. Operators and Hamiltonians are stored as
dense matrices.

# Key Features

- **Full Hilbert space**: All 2^n basis states explicitly represented
- **Dense matrices**: Operators returned as Matrix{ComplexF64}
- **Direct diagonalization**: Standard dense eigenvalue decomposition
- **Simple and straightforward**: No symmetry exploitation

# Quick Start

```julia
using QMBToolkitED

# Get annihilation operators
c = get_annihilators(3)  # Returns Vector{Matrix{ComplexF64}}

# Build Hamiltonian
H, c = build_hamiltonian("hopping.txt", "coulomb.txt")

# Diagonalize
evals, evecs = solve_hamiltonian(H)
```

# Module Structure

- `Operators.jl` - Fermionic operator construction
- `Hamiltonian.jl` - Hamiltonian building from input files
- `Diagonalization.jl` - Eigenvalue decomposition

# Exported Functions

- `get_annihilators_dense` - Generate fermionic operators
- `build_hamiltonian_dense` - Construct Hamiltonian
- `solve_hamiltonian_dense` - Diagonalize Hamiltonian
"""
module Dense

using LinearAlgebra

include("Operators.jl")
include("Hamiltonian.jl")
include("Diagonalization.jl")
include("Greens.jl")

# Exports
export get_annihilators_dense
export build_hamiltonian_dense
export solve_hamiltonian_dense
export calculate_green_tau_dense
export calculate_green_matsubara_dense

# Eigenvalue decomposition results
export DenseEigenResult
export get_eigenvalues, get_eigenvectors

end # module Dense
