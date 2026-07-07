export PseudoInverse

"""
    PseudoInverse <: AbstractCoefficientMethod

A placeholder for solving for the coefficients using the Moore-Penrose pseudoinverse.
This method is often more stable than direct inversion but can be slower.
"""
struct PseudoInverse <: AbstractCoefficientMethod end

function _solve_coefficients_kernel(V::AbstractMatrix{<:Number}, f_reshaped::AbstractVecOrMat{<:Number},::PseudoInverse)
    return pinv(V) * f_reshaped
end
