"""
QMBToolkitED Test Suite

Test Organization:
- global/: User-facing API tests, including consistency validation between Dense and BlockDense
- blockdense/: BlockDense implementation details
"""

using Test

@testset "QMBToolkitED Tests" begin
    @testset "Global API" begin
        include("global/runtests.jl")
    end

    @testset "BlockDense Implementation" begin
        include("blockdense/runtests.jl")
    end
end
