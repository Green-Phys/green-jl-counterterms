export ESPRIT

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

# Kernel function for ESPRIT - returns discrete-time eigenvalues
function _find_discrete_eigenvalues(f::AbstractArray{<:Number}, t::Vector{<:Real},method::ESPRIT)
    # This function ignores `t` as ESPRIT assumes uniform sampling.
    f_reshaped = reshape(f, size(f, 1), :)
    N = size(f_reshaped, 1)
    D = size(f_reshaped, 2)
    L = Int(floor(method.L_factor * N))

    Y = zeros(ComplexF64, D * (N - L), L + 1)
    for i in 1:(L + 1)
        Y[:, i] = vec(@view f_reshaped[i:(i + N - L - 1),:])
    end

    _, S, W = svd(Y, full=false)
    W = conj(W)

    M_eff = method.M
    if M_eff === nothing
        if method.tolerance isa Float64
            M_eff = sum(S .> method.tolerance * S[1])
            if method.opts.verbose
                @info "Estimated model order M = $M_eff using threshold method with tolerance $(method.tolerance)."
            end
        else
            M_eff = _find_M_with_exp_decay(S)
            if method.opts.verbose
                @info "Estimated model order M = $M_eff using exp_decay method."
            end
        end
    end

    W_M = @view W[:, 1:M_eff]
    W_0 = @view W_M[1:end-1, :]
    W_1 = @view W_M[2:end, :]

    F = W_0 \ W_1
    z = eigvals(F)

    return z, S, M_eff
end