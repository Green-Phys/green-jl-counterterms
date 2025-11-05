export Prony

"""
    Prony <: AbstractExponentMethod

Prony's method for finding exponents.

# Fields
- `opts::CommonOptions`: Common options for the method.
- `M::Int`: The model order (number of exponents to find).
"""
struct Prony <: AbstractExponentMethod
    opts::CommonOptions
    M::Int
end

# Constructor to handle keyword arguments
function Prony(; M::Int, verbose::Bool=false)
    if M <= 0
        throw(ArgumentError("Model order M must be a positive integer."))
    end
    opts = CommonOptions(verbose=verbose)
    return Prony(opts, M)
end

# Kernel function for Prony's method - returns discrete-time eigenvalues
function _find_discrete_eigenvalues(method::Prony, f::AbstractArray{<:Complex}, t::Vector{<:Real})
    error("Prony's method is not yet implemented.")
    # The return signature must be: (z, S, M_est)
    # z:     Vector of discrete-time eigenvalues
    # S:     Singular values (return `nothing` as Prony doesn't use SVD)
    # M_est: The estimated model order (should be method.M here)
end