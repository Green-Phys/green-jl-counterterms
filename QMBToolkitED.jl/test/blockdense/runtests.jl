"""
BlockDense Implementation Tests

Tests for BlockDense-specific implementation details.
These verify internal mechanisms like quantum numbers, sectors, block structure, etc.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra

include("test_core.jl")
include("test_algebra.jl")
