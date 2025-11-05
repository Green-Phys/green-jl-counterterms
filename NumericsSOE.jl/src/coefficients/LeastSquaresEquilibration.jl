export LeastSquaresEquilibration
struct LeastSquaresEquilibration <: AbstractCoefficientMethod end

function _solve_coefficients_kernel(V::AbstractMatrix{<:Number}, f_reshaped::AbstractVecOrMat{<:Number}, ::LeastSquaresEquilibration)
    V_max = diagm(vec(maximum(abs.(V), dims=1)))

    V_0 = V * inv(V_max)

    R_0 = V_0 \ f_reshaped

    R = inv(V_max) * R_0
    return R
end
