export LeastSquares

struct LeastSquares <: AbstractCoefficientMethod end

function _solve_coefficients_kernel(V::AbstractMatrix{<:Number}, f_reshaped::AbstractVecOrMat{<:Number},::LeastSquares)
    return V \ f_reshaped
end
