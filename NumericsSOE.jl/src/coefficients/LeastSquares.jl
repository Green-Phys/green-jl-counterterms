export LeastSquares, solve_coefficients

struct LeastSquares <: AbstractCoefficientMethod end

"""
    solve_coefficients(::LeastSquares, s, f, t)

Find the amplitudes `R` for a given set of exponents `s` and a signal `f` sampled at times `t`,
by solving the linear least-squares problem `V * R = f`, where `V` is the Vandermonde matrix.
"""
function solve_coefficients(::LeastSquares, s::Vector{<:Complex}, f::AbstractArray{<:Complex}, t::Vector{<:Real})
    N = size(f, 1)
    original_dim = size(f)[2:end]
    f_reshaped = reshape(f, N, :)

    # The model is f(t) = sum(R_j * exp(s_j * t))
    # This is a linear system V*R = f where V_ij = exp(s_j * t_i)
    V = exp.(t * transpose(s))

    # Solve the least squares problem V * R = f
    R_reshaped = V \ f_reshaped

    # Reshape R back to its original tensor dimensions
    if !isempty(original_dim)
        R = reshape(R_reshaped, (length(s), original_dim...))
    else
        R = R_reshaped
    end

    return R
end
