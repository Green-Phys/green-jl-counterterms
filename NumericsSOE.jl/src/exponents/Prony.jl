export Prony, PronyOutput

struct PronyOutput{T_z, T_S} <: AbstractExponentsOutput
    z::T_z
    S::T_S
end

struct Prony <: AbstractExponentMethod
    opts::CommonOptions
    M::Union{Int, Nothing}
    tolerance::Union{Float64, Nothing}
    L_factor::Float64
end

function Prony(; M=nothing, tolerance=nothing, L_factor=0.4, verbose::Bool=false)
    if !(1/3 <= L_factor <= 1/2)
        throw(ArgumentError("L_factor must be in the range [1/3, 1/2]."))
    end
    if M isa Int && M <= 0
        throw(ArgumentError("Model order M must be a positive integer."))
    end
    if tolerance isa Float64 && tolerance <= 0
        throw(ArgumentError("Tolerance must be a positive number."))
    end
    opts = CommonOptions(verbose=verbose)
    return Prony(opts, M, tolerance, L_factor)
end

# Kernel function for Prony method with principal component regression for denoising.
function _find_discrete_eigenvalues(f::AbstractArray{<:Number}, t::Vector{<:Real},method::Prony)
    # This function ignores `t` as Prony assumes uniform sampling.
    f_reshaped = _prepare_signal_for_hankel(f)

    Y = _hankel_matrix(f_reshaped, method.L_factor)
    Y1 = Y[:, 1:end-1]
    b = -1.0.*Y[:, end]

    U, S, V = svd(Y1, full=false)

    M_eff = _estimate_model_order(method.M, method.tolerance, method.opts.verbose, S)

    # check that M_eff is less than the number of singular values
    if M_eff >= length(S)
        throw(ArgumentError("Estimated model order M = $M_eff is not less than the number of singular values $(length(S))."))
    end

    # Solve the least squares problem using the truncated SVD components.
    # This is equivalent to a = pinv(Y1_M) * b but more numerically stable.
    # The vector `a` stores the coefficients (a_L, ..., a_1) of the polynomial
    Y1_M = U[:, 1:M_eff] * Diagonal(S[1:M_eff]) * V[:, 1:M_eff]'
    a = Y1_M \ b

    z_all = _find_roots_from_companion_matrix(a)

    # pick the eigenvals corresponding to the M_eff with largest magnitudes
    z = sort(z_all, by=abs, rev=true)[1:M_eff]

    return PronyOutput(z, S)
end