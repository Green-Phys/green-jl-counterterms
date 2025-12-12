using QMBToolkitDLR
using Test

@testset "QMBToolkitDLR.jl" begin
    # Dense operations tests
    include("test_dense.jl")

    # BlockDense operations tests
    include("test_blockdense.jl")
end
