module NumericsSOE

using LinearAlgebra
using Statistics

include("common.jl")
include("utils.jl")

# Exponent Methods
include("exponents/ESPRIT.jl")
include("exponents/Prony.jl")

# Coefficient Methods
include("coefficients/LeastSquares.jl")
include("coefficients/PseudoInverse.jl")

export solve, find_exponents, solve_coefficients, ESPRIT, LeastSquares, Prony, PseudoInverse

# Wrapper function with common logic
function find_exponents(method::AbstractExponentMethod, f::AbstractArray{<:Complex}, t::Vector{<:Real})
    # 1. Common Pre-processing
    _check_uniform_spacing(t, method.opts.verbose)
    dt = mean(diff(t))

    # 2. Dispatch to the method-specific kernel
    z, S, M_est = _find_discrete_eigenvalues(method, f, t)

    # 3. Common Post-processing
    s = log.(z) / dt
    
    return s, S, M_est
end

function solve(f::AbstractArray{<:Complex}, t::Vector{<:Real}, exponent_method::AbstractExponentMethod, coefficient_method::AbstractCoefficientMethod)
    s, S, M_est = find_exponents(exponent_method, f, t)
    R = solve_coefficients(coefficient_method, s, f, t)
    return s, R, S, M_est
end

end # module NumericsSOE