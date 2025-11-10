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

    _, S, V = svd(Y, full=false) # Y = U * Diagonal(S) * V'

    M_eff = _estimate_model_order(method.M, method.tolerance, method.opts.verbose, S)

    # check that M_eff is less than the number of singular values
    if M_eff >= length(S)
        throw(ArgumentError("Estimated model order M = $M_eff is not less than the number of singular values $(length(S))."))
    end

    V_M = @view V[:, 1:M_eff]
    V_1 = @view V_M[1:end-1, :]
    V_2 = @view V_M[2:end, :]

    F = V_2' / V_1'  # Solve V_2' = F * V_1' for F
    @assert size(F, 1) == M_eff && size(F, 2) == M_eff
    z = eigvals(F)

    return ESPRITOutput(z, S)
end