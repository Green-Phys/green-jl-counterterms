export ESPRIT, ESPRITOutput

struct ESPRITOutput{T_z, T_S} <: AbstractExponentsOutput
    z::T_z
    S::T_S
end

struct ESPRIT <: AbstractExponentMethod
    opts::CommonOptions
    M::Union{Int, Nothing}
    tolerance::Union{Float64, Nothing}
    L_factor::Float64
end

function ESPRIT(; M=nothing, tolerance=nothing, L_factor=0.4, verbose::Bool=false)
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
    return ESPRIT(opts, M, tolerance, L_factor)
end

# Kernel function for ESPRIT 
function _find_discrete_eigenvalues(f::AbstractArray{<:Number}, t::Vector{<:Real},method::ESPRIT)
    # This function ignores `t` as ESPRIT assumes uniform sampling.
    f_reshaped = _prepare_signal_for_hankel(f)

    Y = _hankel_matrix(f_reshaped, method.L_factor)

    _, S, W = svd(Y, full=false)
    W_conj = conj(W)

    M_eff = _estimate_model_order(method.M, method.tolerance, method.opts.verbose, S)

    # check that M_eff is less than the number of singular values
    if M_eff >= length(S)
        throw(ArgumentError("Estimated model order M = $M_eff is not less than the number of singular values $(length(S))."))
    end

    W_M = @view W_conj[:, 1:M_eff]
    W_0 = @view W_M[1:end-1, :]
    W_1 = @view W_M[2:end, :]

    F = W_0 \ W_1
    z = eigvals(F)

    return ESPRITOutput(z, S)
end