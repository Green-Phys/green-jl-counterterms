module NumericsESPRIT

using LinearAlgebra
using Statistics
export esprit, reconstruct_signal

function esprit(f::AbstractArray{<:Complex}, t::Vector{<:Real}, M::Union{Int, Nothing} = nothing, L_factor::Float64=0.4, tolerance::Float64=1e-15, verbose::Bool=false)
    # check if t is uniformly spaced
    dt = diff(t)
    if maximum(dt) - minimum(dt) > 1e-14
        if verbose
            @warn "t is not uniformly spaced, detecting max(dt) - min(dt) = $(maximum(dt) - minimum(dt))"
        end
    end
    z, R = esprit_original(f, M, L_factor, tolerance)
    s = log.(z) / mean(dt)
    R = R.*exp.(-s * t[1])
    return s, R
end

function reconstruct_signal(s::Vector{<:Complex}, R::AbstractArray{<:Complex}, t::Vector{<:Real})
    N = length(t)
    M = length(s)
    f_rec = zeros(ComplexF64, (N,size(R)[2:end]...))
    for i in 1:N
        selectdim(f_rec, 1, i) .= sum(R[j] .* exp.(s[j] * t[i]) for j in 1:M)
    end
    return f_rec
end

function esprit_original(f::AbstractArray{<:Complex}, M::Union{Int, Nothing} = nothing, L_factor::Float64=0.4, tolerance::Float64=1e-15)
    if !(1.0/3.0 <= L_factor <= 1.0/2.0)
        @warn "For best results, L_factor should be in [1/3, 1/2]"
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
        Y[:, i] = vec(f[i:(i + N - L - 1),:])
    end

    # Perform SVD
    _, S, W = svd(Y)
    W = conj(W)

    # Extract subspaces
    W_M = @view W[:, 1:M]
    W_0 = @view W_M[1:end-1, :]
    W_1 = @view W_M[2:end, :]

    # Solve for the shift operator, using least square fit F = pinv(transpose(W_0)) * transpose(W_1)
    F = W_0 \ W_1

    # Find z
    z = eigvals(F)

    # Find R by solving the linear system
    V = zeros(ComplexF64, N, length(z))
    for i in 1:N
        V[i, :] = z.^(i - 1)
    end
    R = V \ f

    # reshape R to match the input f dimensions
    if length(original_dim) > 0
        R = reshape(R, (length(z), original_dim...))
    end

    return z, R
end

end # module NumericsESPRIT
