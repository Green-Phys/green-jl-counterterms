"""
Validation Test Runner

Compares block-sparse implementation against original full-sparse QMBToolkitED.jl
"""

# Load the BlockSparse module
include("../../src/BlockSparse/BlockSparse.jl")

# Run validation tests
println("="^70)
println("Running Validation Tests: Block-Sparse vs Full-Sparse")
println("="^70)
println()

include("test_validation.jl")
println()

println("="^70)
println("Validation Complete!")
println("="^70)
