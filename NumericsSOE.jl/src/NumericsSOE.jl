module NumericsSOE

using LinearAlgebra
using Statistics
export esprit, reconstruct_signal, esprit_original

"""
    esprit(f::AbstractArray{<:Complex}, t::Vector{<:Real}, M::Union{Int, Nothing} = nothing, tolerance::Union{Float64, Nothing}=nothing, L_factor::Float64=0.4, verbose::Bool=false)

Estimate the signal parameters `s` and amplitudes `R` from a signal `f` sampled at times `t`.

The signal is modeled as a sum of complex exponentials:
`f(t) = sum(R[j] * exp(s[j] * t) for j in 1:M)`

This function is a wrapper around `esprit_original` that handles potentially non-uniformly spaced time `t` and converts the discrete-time eigenvalues `z` to continuous-time parameters `s`.

# Arguments
- `f`: The input signal, a complex-valued array. The first dimension is time.
- `t`: The time points at which the signal `f` is sampled.
- `M`: The number of exponential components to estimate (model order). If not provided, it is estimated automatically.
- `tolerance`: If `M` is `nothing`, this tolerance is used for the `:threshold` method of `M` estimation. If `tolerance` is also `nothing`, a knee-finding algorithm is used.
- `L_factor`: A factor to determine the size of the covariance matrix, should be in [1/3, 1/2]. Defaults to 0.4.
- `verbose`: If true, prints warnings and info messages.

# Returns
- `s`: A vector of complex numbers representing the estimated signal parameters (frequencies and damping factors).
- `R`: An array of complex numbers representing the amplitudes of each component.
- `S`: The singular values of the Hankel matrix.
- `M_est`: The estimated (or provided) model order.
"""
function esprit(f::AbstractArray{<:Complex}, t::Vector{<:Real}, M::Union{Int, Nothing} = nothing, tolerance::Union{Float64, Nothing}=nothing, L_factor::Float64=0.4, verbose::Bool=false)
    # check if t is uniformly spaced
    dt = diff(t)
    if maximum(dt) - minimum(dt) > 1e-14
        if verbose
            @warn "t is not uniformly spaced, detecting max(dt) - min(dt) = $(maximum(dt) - minimum(dt))"
        end
    end
    z, R, S, M_est = esprit_original(f, M, tolerance, L_factor, verbose)
    s = log.(z) / mean(dt)
    R = R.*exp.(-s * t[1])
    return s, R, S, M_est
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

function find_knee(y::Vector{Float64})
    n = length(y)
    if n < 2
        return n
    end
    x = 1:n
    
    # Points defining the line
    p1 = [Float64(x[1]), y[1]]
    p2 = [Float64(x[end]), y[end]]
    
    # Vector for the line
    line_vec = p2 - p1
    
    line_vec_norm_sq = sum(line_vec.^2)
    if line_vec_norm_sq == 0
        return 1
    end

    # Find distances
    distances = zeros(n)
    for i in 2:(n-1)
        p_i = [Float64(x[i]), y[i]]
        p1_to_p_i = p_i - p1
        area = abs(line_vec[1] * p1_to_p_i[2] - line_vec[2] * p1_to_p_i[1])
        distances[i] = area
    end
    
    knee_idx = argmax(distances)
    if distances[knee_idx] == 0
        return 1 # All points are on the line
    end

    return knee_idx
end

function find_M_with_exp_decay(S::Vector{Float64})
    if isempty(S)
        return 0
    end
    
    logS = log.(S)
    
    # Find the knee of the log-plot of singular values
    knee = find_knee(logS)
    
    if knee < 2
        return knee
    end
    
    # Take S up to the knee
    S_truncated = S[1:knee]
    
    # Calculate the derivative of the log of these singular values
    dlogS = abs.(diff(log.(S_truncated)))
    
    if isempty(dlogS)
        return 1
    end
    
    # Find where the derivative is significant
    threshold = maximum(dlogS) / 3.0
    indices = findall(dlogS .> threshold)
    
    if isempty(indices)
        return 1 # Default to 1 if no significant drop is found
    end
    
    M = last(indices)
    return M
end

function esprit_original(f::AbstractArray{<:Complex}, M::Union{Int, Nothing} = nothing, tolerance::Union{Float64, Nothing}=nothing, L_factor::Float64=0.4, verbose::Bool=false)
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
    _, S, W = svd(Y, full=false) # In julia, svd returns U, S, W such that Y = U * Diagonal(S) * W
    W = conj(W)

    M_eff = M
    if M_eff === nothing
        if tolerance isa Float64
            M_eff = sum(S .> tolerance * S[1])
            if verbose
                @info "Estimated model order M = $M_eff using threshold method with tolerance $tolerance."
            end
        else
            M_eff = find_M_with_exp_decay(S)
            if verbose
                @info "Estimated model order M = $M_eff using exp_decay method."
            end
        end
    end

    # Extract subspaces
    W_M = @view W[:, 1:M_eff]
    W_0 = @view W_M[1:end-1, :]
    W_1 = @view W_M[2:end, :]

    # Solve for the shift operator, using least square fit F = pinv(W_0) * W_1 
    F = W_0 \ W_1

    # Find z
    z = eigvals(F)

    # Find R by solving the linear system
    V = transpose(z).^(0:N-1)
    # R = inv(V' * V) * V' * f # explicit pseudo-inverse solution
    # R = pinv(V) * f # using pinv
    # R = V \ f # lsq solution

    # V_max = diagm(vec(maximum(abs.(V), dims=1))) # scale V to improve conditioning
    # V_0 = V * inv(V_max)
    # R = inv(V_max) * inv(V_0' * V_0) * V_0' * f

    V_max = diagm(vec(maximum(abs.(V), dims=1)))
    V_0 = V * inv(V_max)
    R_0 = V_0 \ f
    R = inv(V_max) * R_0

    # reshape R to match the input f dimensions
    if length(original_dim) > 0
        R = reshape(R, (length(z), original_dim...))
    end

    return z, R, S, M_eff
end

end # module NumericsSOE
