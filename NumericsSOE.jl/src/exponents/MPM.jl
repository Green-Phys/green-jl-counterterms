export MPM, MPMOutput

struct MPMOutput{T_z, T_S} <: AbstractExponentsOutput
    z::T_z
    S::T_S
end

struct MPM <: AbstractExponentMethod
    opts::CommonOptions
    M::Union{Int, Nothing}
    tolerance::Union{Float64, Nothing}
    L_factor::Float64
end

function MPM(; M=nothing, tolerance=nothing, L_factor=0.4, verbose::Bool=false)
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
    return MPM(opts, M, tolerance, L_factor)
end

# Kernel function for MPM 
function _find_discrete_eigenvalues(f::AbstractArray{<:Number}, t::Vector{<:Real},method::MPM)
    # This function ignores `t` as MPM assumes uniform sampling.
    f_reshaped = _prepare_signal_for_hankel(f)

    Y = _hankel_matrix(f_reshaped, method.L_factor)

    U, S, V = svd(Y, full=false) # Y = U * Diagonal(S) * V'

    M_eff = _estimate_model_order(method.M, method.tolerance, method.opts.verbose, S)

    # check that M_eff is less than the number of singular values
    if M_eff >= length(S)
        throw(ArgumentError("Estimated model order M = $M_eff is not less than the number of singular values $(length(S))."))
    end

    Y_M = U[:, 1:M_eff] * Diagonal(S[1:M_eff]) * V[:, 1:M_eff]'
    Y1 = Y_M[:, 1:end-1]
    Y2 = Y_M[:, 2:end]
    F = Y2 / Y1  # Solve Y2 = F * Y1 for F
    z = eigvals(F)
    z = sort(z, by=abs, rev=true)[1:M_eff]

    return MPMOutput(z, S)
end