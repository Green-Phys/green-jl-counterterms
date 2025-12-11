"""
Global Test Suite

Tests for user-facing API - works with both Dense and BlockDense methods.
These tests verify that the package functions correctly from a user's perspective.
"""

using Test

include("test_utils.jl")
using .GlobalTestUtils

include("test_operators.jl")
include("test_hamiltonians.jl")
include("test_greens.jl")
include("test_unified_interface.jl")
include("test_performance.jl")
