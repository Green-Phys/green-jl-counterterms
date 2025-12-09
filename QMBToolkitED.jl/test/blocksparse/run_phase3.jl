"""
Phase 3 Test Runner

Runs all Phase 3 tests for BlockSparse linear algebra operations.
"""

# Load the BlockSparse module
include("../../src/BlockSparse/BlockSparse.jl")

# Run Phase 3 tests
println("="^70)
println("Running Phase 3: Linear Algebra Tests")
println("="^70)
println()

include("test_linalg.jl")
println()

println("="^70)
println("Phase 3 Complete!")
println("="^70)
