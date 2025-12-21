"""
BlockDense module for quantum many-body exact diagonalization with symmetry blocking.

This module implements a block-dense linear algebra library that exploits quantum number
conservation to reduce memory usage and computational cost by 10-100x compared to
full Hilbert space methods.

# Key Features

- **Site-based quantum numbers**: Specify quantum numbers for each orbital
- **Automatic sector generation**: States grouped by total quantum number
- **Block-dense operators**: Only non-zero blocks stored as dense matrices
- **Flexible quantum numbers**: Supports (N, Sz), N-only, and extensible to others

# Quick Start

```julia
using QMBToolkitED.BlockDense

# Define a 4-flavor Hubbard atom with (N, Sz) conservation
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
basis = make_basis(site_qns)

# Get annihilation operators
c = get_annihilators_blockdense(basis)

# Build number operator
n_up = c[1]' * c[1]
```

# Module Structure

## Core/ - Core types and data structures
- `QuantumNumbers.jl` - Abstract QuantumNumber type, QN_NSz, QN_N
- `Sectors.jl` - Sector type for grouping states
- `Basis.jl` - BlockDenseBasis and make_basis function
- `Operators.jl` - BlockDenseOperator type and single-operator construction

## Operations/ - Low-level operations on operators
- `Algebra.jl` - Operator algebra (*, +, adjoint, scalar operations)
- `Conversion.jl` - Conversion to/from dense matrices

## HighLevel/ - High-level API matching QMBToolkitED interface
- `Constructors.jl` - get_annihilators_blockdense (vector of operators)
- `Hamiltonian.jl` - build_hamiltonian_blockdense
- `Diagonalization.jl` - solve_hamiltonian_blockdense with efficient block diagonalization

## Utils.jl - Utility functions
- sum_occupied_sites, compute_fermion_sign, etc.

# Exported Types

- `QuantumNumber` - Abstract type for quantum numbers
- `QN_NSz` - (N, Sz) quantum number
- `QN_N` - N-only quantum number
- `Sector` - Group of states with same quantum number
- `BlockDenseBasis` - Full basis partitioned into sectors
- `OperatorFlux` - Quantum number change carried by operator
- `BlockDenseOperator` - Block-dense matrix operator
- `BlockEigenResult` - Result of block-diagonal eigenvalue decomposition

# Exported Functions

## Core functions
- `make_basis` - Construct basis from site quantum numbers
- `get_annihilator_blockdense` - Create single annihilation operator
- `combine_qn`, `reverse_qn`, `zero_qn` - Quantum number arithmetic
- `sum_occupied_sites`, `compute_fermion_sign` - Utility functions

## High-level API (compatible with QMBToolkitED)
- `get_annihilators_blockdense` - Generate vector of annihilation operators
- `build_hamiltonian_blockdense` - Construct Hamiltonian from input files
- `solve_hamiltonian_blockdense` - Efficient block-diagonal diagonalization

## Conversion
- `to_matrix`, `from_matrix` - Convert between block-dense and dense matrix formats

## Eigenvalue decomposition utilities
- `get_full_eigenvalues` - Extract all eigenvalues from BlockEigenResult
- `get_full_eigenvectors` - Extract all eigenvectors in global basis
"""
module BlockDense

using LinearAlgebra
import ..Dense: to_matrix

# ============================================================================
# Core Types and Data Structures
# ============================================================================

include("Core/QuantumNumbers.jl")
include("Core/Sectors.jl")
include("Core/Basis.jl")
include("Core/Operators.jl")

# ============================================================================
# Utilities
# ============================================================================

include("Utils.jl")

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

# Quantum Numbers
export QuantumNumber, QN_NSz, QN_N
export combine_qn, reverse_qn, zero_qn

# Sectors and Basis
export Sector, BlockDenseBasis, make_basis

# Operators
export OperatorFlux, BlockDenseOperator
export get_annihilator_blockdense  # Single operator (low-level)

# Utilities
export sum_occupied_sites, compute_fermion_sign

# Conversion
export to_matrix, from_matrix

# High-Level API (compatible with QMBToolkitED)
export get_annihilators_blockdense        # Vector of operators
export build_hamiltonian_blockdense        # Hamiltonian construction
export solve_hamiltonian_blockdense        # Standard eigenvalue solver
export calculate_green_tau_blockdense      # Green's function (imaginary time)
export calculate_green_matsubara_blockdense # Green's function (Matsubara)

# Eigenvalue decomposition results
export BlockEigenResult
export get_full_eigenvalues, get_full_eigenvectors

end # module BlockDense
