"""
QMBToolkitED Test Suite

Test Organization:
- physics/: Physics validation tests against analytical results
- unit/: Unit tests for types, basis, operators, and algebra
"""

using Test

@testset "QMBToolkitED Tests" begin
    @testset "Physics Validation" begin
        include("physics/runtests.jl")
    end

    @testset "Unit Tests" begin
        include("unit/runtests.jl")
    end
end
