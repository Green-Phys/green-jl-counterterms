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

# Export conversion functions (from BlockDense module)
export to_dense, from_dense

# Export submodules for direct access
export Dense, BlockDense

# Export Solver Result Types (from UnifiedInterface)
export DenseSolverResult, BlockEigenResult

end # module QMBToolkitED