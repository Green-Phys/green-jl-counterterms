"""
QMBToolkitED Test Suite

Test Organization:
- global/: User-facing API tests (works with both Dense and BlockDense)
- blockdense/: BlockDense implementation details
- consistency/: Validates Dense and BlockDense produce identical results
"""

using Test

@testset "QMBToolkitED Tests" begin
    @testset "Global API" begin
        include("global/runtests.jl")
    end

    @testset "BlockDense Implementation" begin
        include("blockdense/runtests.jl")
    end

    @testset "Consistency" begin
        include("consistency/runtests.jl")
    end
end
