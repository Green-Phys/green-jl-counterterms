using NumericsSOE
using Test

# Include shared test utilities
include("test_utils.jl")

@testset "NumericsSOE.jl" begin
    include("esprit_tests.jl")
    include("prony_tests.jl")
    # include("prony_tls_tests.jl")
end
