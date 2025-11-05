export PseudoInverseEquilibration

"""
    PseudoInverseEquilibration <: AbstractCoefficientMethod

A placeholder for solving for the coefficients using the Moore-Penrose PseudoInverseEquilibration.
This method is often more stable than direct inversion but can be slower.
"""
struct PseudoInverseEquilibration <: AbstractCoefficientMethod end

function _solve_coefficients_kernel(V::AbstractMatrix{<:Number}, f_reshaped::AbstractVecOrMat{<:Number},::PseudoInverseEquilibration)
    V_max = diagm(vec(maximum(abs.(V), dims=1)))

    V_0 = V * inv(V_max)

    R_0 = pinv(V_0) * f_reshaped

    R = inv(V_max) * R_0
    return R
end
