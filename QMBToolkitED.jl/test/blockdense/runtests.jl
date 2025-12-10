"""
BlockDense Implementation Tests

Tests for BlockDense-specific implementation details.
These verify internal mechanisms like quantum numbers, sectors, block structure, etc.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra
using SparseArrays

include("test_core.jl")
include("test_algebra.jl")
