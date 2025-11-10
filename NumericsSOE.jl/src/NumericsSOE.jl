module NumericsSOE

using LinearAlgebra
using Statistics

include("common.jl")
include("utils.jl")

# Exponent Methods
include("exponents/ESPRIT.jl")
include("exponents/Prony.jl")
include("exponents/PronyTLS.jl") ## TODO: investigate this
include("exponents/MPM.jl")

# Coefficient Methods
include("coefficients/LeastSquares.jl")
include("coefficients/PseudoInverse.jl")
include("coefficients/LeastSquaresEquilibration.jl")
include("coefficients/PseudoInverseEquilibration.jl")

export solve, find_exponents, solve_coefficients, SOEOutput
export Prony, ESPRIT, MPM
export LeastSquares, PseudoInverse, LeastSquaresEquilibration, PseudoInverseEquilibration


# Wrapper function with common logic
function find_exponents(f::AbstractArray{<:Number}, t::Vector{<:Real}, method::AbstractExponentMethod=ESPRIT())
    # 1. Common Pre-processing
    _check_uniform_spacing(t, method.opts.verbose)
    dt = mean(diff(t))

    # 2. Dispatch to the method-specific kernel
    exponents_output = _find_discrete_eigenvalues(f, t, method)

    # 3. Common Post-processing
    s = log.(exponents_output.z) / dt
    
    return s, exponents_output
end

function solve_coefficients(s::Vector{<:Number}, f::AbstractArray{<:Number}, t::Vector{<:Real}, method::AbstractCoefficientMethod=LeastSquares())
    original_dim = size(f)[2:end]
    f_reshaped = reshape(f, size(f, 1), :)

    # The model is f(t) = sum(R_j * exp(s_j * t))
    # This is a linear system V*R = f where V_ij = exp(s_j * t_i)
    V = exp.(t * transpose(s))

    # Solve the least squares problem V * R = f
    R_reshaped = _solve_coefficients_kernel(V, f_reshaped, method)

    # Reshape R back to its original tensor dimensions
    if !isempty(original_dim)
        R = reshape(R_reshaped, (length(s), original_dim...))
    else
        R = R_reshaped
    end

    return R
end

function solve(f::AbstractArray{<:Number}, t::Vector{<:Real}, exponent_method::AbstractExponentMethod=ESPRIT(), coefficient_method::AbstractCoefficientMethod=LeastSquares())
    s, exponents_output = find_exponents(f, t, exponent_method)
    R = solve_coefficients(s, f, t, coefficient_method)
    return SOEOutput(s, R, exponents_output)
end

end # module NumericsSOE