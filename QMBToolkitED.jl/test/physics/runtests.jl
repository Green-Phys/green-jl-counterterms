"""
Physics Test Suite

Tests for physical models against analytical solutions.
Validates correctness with QN_Trivial, QN_N, and QN_NSz quantum numbers.
"""

using Test

include("test_utils.jl")
using .PhysicsTestUtils

for model in physics_models()
    @testset "$(model.name)" begin
        run_physics_test(model)
    end
end
