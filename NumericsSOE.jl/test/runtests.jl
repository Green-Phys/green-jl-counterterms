using NumericsSOE
using Test

# Include shared test utilities
include("test_utils.jl")

@testset "NumericsSOE.jl" begin
    include("prony_tests.jl")
    include("esprit_tests.jl")
    include("mpm_tests.jl")
    # include("prony_tls_tests.jl")
end
