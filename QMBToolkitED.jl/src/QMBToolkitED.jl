module QMBToolkitED

using LinearAlgebra

# Include dense submodule
include("Dense/Dense.jl")
using .Dense

# Include block-dense submodule
include("BlockDense/BlockDense.jl")
using .BlockDense

# Include unified interface (provides consistent API)
include("UnifiedInterface.jl")

# Exports - main functions (from unified interface)
export get_annihilators, build_hamiltonian, solve_hamiltonian
export calculate_green_tau, calculate_green_matsubara, get_Z

# Export conversion functions
# to_matrix is defined in Dense and extended by BlockDense
# from_matrix is defined in BlockDense only
export to_matrix, from_matrix

# Export submodules for direct access
export Dense, BlockDense

# Export Solver Result Types
export DenseEigenResult, BlockEigenResult

# Export Core Types for convenience
export DenseBasis, DenseOperator, DenseDiagonalOperator
export BlockDenseBasis, BlockDenseOperator, BlockDenseDiagonalOperator

end # module QMBToolkitED
