"""
Dense module for quantum many-body exact diagonalization using full Hilbert space.

This module implements the traditional full Hilbert space approach, where all 2^n
basis states are explicitly represented. Operators and Hamiltonians are stored as
dense matrices wrapped in DenseOperator structs.

# Key Features

- **Full Hilbert space**: All 2^n basis states explicitly represented
- **Dense matrices**: Operators stored as DenseOperator wrapping Matrix{ComplexF64}
- **Direct diagonalization**: Standard dense eigenvalue decomposition
- **Simple and straightforward**: No symmetry exploitation

# Quick Start

```julia
using QMBToolkitED

# Create basis and get annihilation operators
basis = make_basis_dense(3)
c = get_annihilators_dense(basis)

# Or use the convenience method
c = get_annihilators_dense(3)

# Build Hamiltonian
H, c = build_hamiltonian_dense("hopping.txt", "coulomb.txt")

# Diagonalize
result = solve_hamiltonian_dense(H)
evals, evecs = result  # Destructuring works
```

# Module Structure

## Core/ - Core types and data structures
- `Basis.jl` - DenseBasis struct
- `Operators.jl` - DenseOperator struct

## Operations/ - Operations on operators
- `Algebra.jl` - Operator algebra (*, +, adjoint, scalar operations)

## HighLevel/ - High-level API matching QMBToolkitED interface
- `Constructors.jl` - Fermionic operator construction
- `Hamiltonian.jl` - Hamiltonian building from input files
- `Diagonalization.jl` - Eigenvalue decomposition
- `Greens.jl` - Green's function calculations

# Exported Types

- `DenseBasis` - Full Hilbert space basis
- `DenseOperator` - Dense matrix operator with basis
- `DenseEigenResult` - Result of eigenvalue decomposition

# Exported Functions

- `make_basis_dense` - Construct dense basis
- `get_annihilators_dense` - Generate fermionic operators
- `build_hamiltonian_dense` - Construct Hamiltonian
- `solve_hamiltonian_dense` - Diagonalize Hamiltonian
- `calculate_green_tau_dense` - Compute G(τ)
- `calculate_green_matsubara_dense` - Compute G(iωₙ)
- `to_matrix` - Extract underlying matrix from DenseOperator
"""
module Dense

using LinearAlgebra

# ============================================================================
# Core Types and Data Structures
# ============================================================================

include("Core/Basis.jl")
include("Core/Operators.jl")

# ============================================================================
# Operations on Operators
# ============================================================================

include("Operations/Algebra.jl")
include("Operations/Conversion.jl")

# ============================================================================
# High-Level API
# ============================================================================

include("HighLevel/Constructors.jl")
include("HighLevel/Hamiltonian.jl")
include("HighLevel/Diagonalization.jl")
include("HighLevel/Greens.jl")

# ============================================================================
# Exports
# ============================================================================

# Core types
export DenseBasis, make_basis_dense
export DenseOperator, to_matrix

# High-level API
export get_annihilators_dense
export build_hamiltonian_dense
export solve_hamiltonian_dense
export calculate_green_tau_dense
export calculate_green_matsubara_dense

# Eigenvalue decomposition results
export DenseEigenResult
export get_eigenvalues, get_eigenvectors

end # module Dense
