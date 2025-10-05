module NumericsESPRIT

using LinearAlgebra
using Statistics
export esprit, reconstruct_signal

"""
    esprit(f::AbstractArray{<:Complex}, t::Vector{<:Real}, M::Union{Int, Nothing} = nothing, L_factor::Float64=0.4, tolerance::Float64=1e-15, verbose::Bool=false)

Estimate the signal parameters `s` and amplitudes `R` from a signal `f` sampled at times `t`.

The signal is modeled as a sum of complex exponentials:
`f(t) = sum(R[j] * exp(s[j] * t) for j in 1:M)`

This function is a wrapper around `esprit_original` that handles potentially non-uniformly spaced time `t` and converts the discrete-time eigenvalues `z` to continuous-time parameters `s`.

# Arguments
- `f`: The input signal, a complex-valued array. The first dimension is time.
- `t`: The time points at which the signal `f` is sampled.
- `M`: The number of exponential components to estimate (model order). If set to `nothing`, the model order is estimated automatically based on the `tolerance`.
- `L_factor`: A factor to determine the size of the covariance matrix, should be in [1/3, 1/2]. Defaults to 0.4.
- `tolerance`: Tolerance for the automatic estimation of the model order `M`. It is relative to the largest singular value. Defaults to 1e-15.
- `verbose`: If true, prints warnings and info messages.

# Returns
- `s`: A vector of complex numbers representing the estimated signal parameters (frequencies and damping factors).
- `R`: An array of complex numbers representing the amplitudes of each component. The dimensions of `R` will be `(M, size(f)[2:end]...)`.
"""
function esprit(f::AbstractArray{<:Complex}, t::Vector{<:Real}, M::Union{Int, Nothing} = nothing, L_factor::Float64=0.4, tolerance::Float64=1e-15, verbose::Bool=false)
    # check if t is uniformly spaced
    dt = diff(t)
    if maximum(dt) - minimum(dt) > 1e-14
        if verbose
            @warn "t is not uniformly spaced, detecting max(dt) - min(dt) = $(maximum(dt) - minimum(dt))"
        end
    end
    z, R = esprit_original(f, M, L_factor, tolerance, verbose)
    s = log.(z) / mean(dt)
    R = R.*exp.(-s * t[1])
    return s, R
end

"""
    reconstruct_signal(s::Vector{<:Complex}, R::AbstractArray{<:Complex}, t::Vector{<:Real})

Reconstruct a signal from its estimated parameters `s` and amplitudes `R` at given time points `t`.

The reconstructed signal is calculated as:
`f_rec(t) = sum(R[j] * exp(s[j] * t) for j in 1:M)`
where `M` is the number of components (`length(s)`).

# Arguments
- `s`: The vector of signal parameters (frequencies and damping factors).
- `R`: The array of amplitudes. The first dimension should correspond to the number of components `M`.
- `t`: The time points at which to reconstruct the signal.

# Returns
- `f_rec`: The reconstructed signal as an array. The first dimension is time.
"""
function reconstruct_signal(s::Vector{<:Complex}, R::AbstractArray{<:Complex}, t::Vector{<:Real})
    N = length(t)
    M = length(s)
    f_rec = zeros(ComplexF64, (N,size(R)[2:end]...))
    for i in 1:N
        selectdim(f_rec, 1, i) .= sum(selectdim(R, 1, j) .* exp(s[j] * t[i]) for j in 1:M)
    end
    return f_rec
end

"""
    esprit_original(f::AbstractArray{<:Complex}, M::Union{Int, Nothing} = nothing, L_factor::Float64=0.4, tolerance::Float64=1e-15, verbose::Bool=false)

The core ESPRIT algorithm implementation.

It estimates the parameters of a signal that is a sum of complex exponentials.
This function assumes a uniformly sampled signal and returns the discrete-time eigenvalues `z` and amplitudes `R`.

# Arguments
- `f`: The input signal, a complex-valued array. The first dimension is assumed to be the uniformly sampled time series.
- `M`: The model order (number of exponential components). If set to `nothing`, the model order is estimated automatically based on the `tolerance`.
- `L_factor`: Factor to determine the size of the Hankel matrix. It is recommended to be in [1/3, 1/2].
- `tolerance`: Tolerance for the automatic estimation of the model order `M`. It is relative to the largest singular value. Defaults to 1e-15.
- `verbose`: If true, prints warnings and info messages.

# Returns
- `z`: The eigenvalues of the shift operator, which are related to the frequencies of the signal components. These are discrete-time frequencies.
- `R`: The amplitudes of the signal components.
"""
function esprit_original(f::AbstractArray{<:Complex}, M::Union{Int, Nothing} = nothing, L_factor::Float64=0.4, tolerance::Float64=1e-15, verbose::Bool=false)
    if !(1.0/3.0 <= L_factor <= 1.0/2.0)
        if verbose
            @warn "L_factor should be in [1/3, 1/2] for best results, got L_factor = $L_factor"
        end
    end

    if ndims(f) !=1
        original_dim = size(f)[2:end]
    else
        original_dim = ()
    end
    # reshape f to 2D
    f = reshape(f, size(f, 1), :)
    N = size(f, 1) # signal length
    D = size(f, 2) # number of signals that share the same frequencies
    L = Int(floor(L_factor * N))

    # Build the Hankel-like matrix Y of size D*(N-L) x L+1
    Y = zeros(ComplexF64, D * (N - L), L + 1)
    for i in 1:(L + 1)
        Y[:, i] = vec(@view f[i:(i + N - L - 1),:])
    end

    # Perform SVD
    _, S, W = svd(Y) # Y = U * Diagonal(S) * W'
    W = conj(W)

    if M === nothing
        M = sum(S .> tolerance * S[1])
        if verbose
            @info "Estimated model order M = $M"
        end
    end

    # Extract subspaces
    W_M = @view W[:, 1:M]
    W_0 = @view W_M[1:end-1, :]
    W_1 = @view W_M[2:end, :]

    # Solve for the shift operator, using least square fit F = pinv(transpose(W_0)) * transpose(W_1)
    F = W_0 \ W_1

    # Find z
    z = eigvals(F)

    # Find R by solving the linear system
    V = transpose(z).^(0:N-1)
    R = V \ f

    # reshape R to match the input f dimensions
    if length(original_dim) > 0
        R = reshape(R, (length(z), original_dim...))
    end

    return z, R
end

end # module NumericsESPRIT