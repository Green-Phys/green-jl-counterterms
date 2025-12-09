# Test runner for Phase 1
# Run from project root: julia --project test/blocksparse/run_phase1.jl

# Load the BlockSparse module
include("../../src/BlockSparse/BlockSparse.jl")
using .BlockSparse

# Run the tests
include("test_types.jl")
