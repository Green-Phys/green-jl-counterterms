"""
BlockSparse module for quantum many-body exact diagonalization with symmetry blocking.

This module implements a block-sparse linear algebra library that exploits quantum number
conservation to reduce memory usage and computational cost by 10-100x compared to
full Hilbert space methods.

# Key Features

- **Site-based quantum numbers**: Specify quantum numbers for each orbital
- **Automatic sector generation**: States grouped by total quantum number
- **Block-sparse operators**: Only non-zero blocks stored as dense matrices
- **Flexible quantum numbers**: Supports (N, Sz), N-only, and extensible to others

# Quick Start

```julia
using QMBToolkitED.BlockSparse

# Define a 4-flavor Hubbard atom with (N, Sz) conservation
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
basis = make_basis(site_qns)

# Get annihilation operators
c = [get_annihilator_blocksparse(basis, i) for i in 1:4]

# Build number operator
n_up = c[1]' * c[1]
```

# Exported Types

- `QuantumNumber` - Abstract type for quantum numbers
- `QN_NSz` - (N, Sz) quantum number
- `QN_N` - N-only quantum number
- `Sector` - Group of states with same quantum number
- `BlockSparseBasis` - Full basis partitioned into sectors
- `OperatorFlux` - Quantum number change carried by operator
- `BlockSparseOperator` - Block-sparse matrix operator

# Exported Functions

- `make_basis` - Construct basis from site quantum numbers
- `get_annihilator_blocksparse` - Create annihilation operator
- `combine_qn` - Add quantum numbers
- `reverse_qn` - Negate quantum number (for adjoint)
- `zero_qn` - Identity element
- `sum_occupied_sites` - Compute total QN from state
- `compute_fermion_sign` - Fermionic sign for operators
"""
module BlockSparse

using LinearAlgebra
using SparseArrays

# Include type definitions
include("QuantumNumbers.jl")
include("Sectors.jl")
include("Utils.jl")
include("Basis.jl")
include("Operators.jl")

# Include operations
include("BlockOperations.jl")
include("Conversion.jl")
include("BlockSparseInterface.jl")

# Exports - Quantum Numbers
export QuantumNumber, QN_NSz, QN_N
export combine_qn, reverse_qn, zero_qn

# Exports - Sectors
export Sector

# Exports - Basis
export BlockSparseBasis, make_basis

# Exports - Operators
export OperatorFlux, BlockSparseOperator
export get_annihilator_blocksparse

# Exports - Utilities
export sum_occupied_sites, compute_fermion_sign

# Exports - Conversion
export to_sparse, from_sparse

# Exports - Interface (API-compatible functions)
export get_annihilators_blocksparse
export build_hamiltonian_blocksparse
export solve_hamiltonian_blocksparse

end # module BlockSparse
