using NumericsSOE
using Test

# Include shared test utilities
include("test_utils.jl")

@testset "NumericsSOE.jl" begin
    include("test_esprit.jl")
end
