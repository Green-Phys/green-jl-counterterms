export PseudoInverse

"""
    PseudoInverse <: AbstractCoefficientMethod

A placeholder for solving for the coefficients using the Moore-Penrose pseudoinverse.
This method is often more stable than direct inversion but can be slower.
"""
struct PseudoInverse <: AbstractCoefficientMethod end

function solve_coefficients(method::PseudoInverse, s::Vector{<:Complex}, f::AbstractArray{<:Complex}, t::Vector{<:Real})
    error("PseudoInverse coefficient solver is not yet implemented.")
    # The return signature should be: R
    # R: An array of coefficients with dimensions (length(s), size(f)[2:end]...)
end
