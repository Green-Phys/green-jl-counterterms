"""
Phase 2 Test Runner

Runs all Phase 2 tests for BlockSparse basis and operators.
"""

# Load the BlockSparse module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
include("../../src/BlockSparse/BlockSparse.jl")

# Run Phase 2 tests
println("="^70)
println("Running Phase 2: Basis and Operators Tests")
println("="^70)
println()

include("test_basis.jl")
println()

include("test_operators.jl")
println()

println("="^70)
println("Phase 2 Complete!")
println("="^70)
