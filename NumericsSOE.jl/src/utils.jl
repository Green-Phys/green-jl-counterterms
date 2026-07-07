export reconstruct_signal

function _kernel_tau(tau, omega, beta)
    tau=Float64(tau)
    omega=Float64(omega)
    beta=Float64(beta)
    @assert tau >= -beta && tau <= beta "tau must be in [-beta, beta]"
    if tau >= 0.0
        if omega > 0.0
            return exp(-tau * omega)/(1 + exp(-beta * omega))
        else
            return exp(-(tau - beta) * omega)/(1 + exp(beta * omega))   
        end
    else
        if omega > 0.0
            return -1 * exp(-(tau + beta) * omega)/(1 + exp(-beta * omega))
        else
            return -1 * exp(-tau * omega)/(1 + exp(beta * omega))   
        end
    end
end

function _estimate_model_order(M::Union{Int, Nothing}, tolerance::Union{Float64, Nothing}, verbose::Bool, S::AbstractVector{<:Real})
    M_eff = M
    if M_eff === nothing
        if tolerance isa Float64
            M_eff = sum(S .> tolerance * S[1])
            if verbose
                @info "Estimated model order M = $M_eff using threshold method with tolerance $tolerance."
            end
        else
            M_eff = _find_M_with_exp_decay(S)
            if verbose
                @info "Estimated model order M = $M_eff using exp_decay method."
            end
        end
    end
    return M_eff
end

# Internal helper to check for uniform time spacing
function _check_uniform_spacing(t::Vector{<:Real}, verbose::Bool)
    if length(t) > 1
        dt = diff(t)
        if maximum(dt) - minimum(dt) > 1e-14 # Use a tolerance for floating point comparisons
            if verbose
                @warn "Time vector `t` is not uniformly spaced. This may affect results for methods assuming uniform spacing."
            end
        end
    end
end

function _prepare_signal_for_hankel(f::AbstractArray{<:Number})
    f_reshaped = reshape(f, size(f, 1), :)  # TODO: there is other ways to arange data for 2+1D signals
    return f_reshaped
end

function _hankel_matrix(f_reshaped::AbstractMatrix{<:Number}, L_factor::Float64)
    N = size(f_reshaped, 1)
    D = size(f_reshaped, 2)
    L = Int(floor(L_factor * N)) 
    Y = zeros(ComplexF64, D * (N - L), L + 1)
    for i in 1:(L + 1)
        Y[:, i] = vec(@view f_reshaped[i:(i + N - L - 1),:])
    end
    return Y
end

"""
    reconstruct_signal(s::Vector{<:Number}, R::AbstractArray{<:Number}, t::Vector{<:Real})

Reconstruct a signal from its estimated parameters `s` and amplitudes `R` at given time points `t`.
"""
function reconstruct_signal(s::Vector{<:Number}, R::AbstractArray{<:Number}, t::Vector{<:Real}; is_Euclidian_Fermi::Bool=false)
    N = length(t)
    M = length(s)
    f_rec = zeros(ComplexF64, (N,size(R)[2:end]...))
    for i in 1:N
        if !is_Euclidian_Fermi
            selectdim(f_rec, 1, i) .= sum(selectdim(R, 1, j) .* exp(s[j] * t[i]) for j in 1:M)
        else
            beta = t[end]
            selectdim(f_rec, 1, i) .= sum(selectdim(R, 1, j) .* _kernel_tau(t[i], -s[j], beta) for j in 1:M)
        end
    end
    return f_rec
end

function _find_knee(y::Vector{Float64})
    n = length(y)
    if n < 2
        return n
    end
    x = 1:n
    
    p1 = [Float64(x[1]), y[1]]
    p2 = [Float64(x[end]), y[end]]
    
    line_vec = p2 - p1
    
    line_vec_norm_sq = sum(line_vec.^2)
    if line_vec_norm_sq == 0
        return 1
    end

    distances = zeros(n)
    for i in 2:(n-1)
        p_i = [Float64(x[i]), y[i]]
        p1_to_p_i = p_i - p1
        area = abs(line_vec[1] * p1_to_p_i[2] - line_vec[2] * p1_to_p_i[1])
        distances[i] = area
    end
    
    knee_idx = argmax(distances)
    if distances[knee_idx] == 0
        return 1
    end

    return knee_idx
end

function _find_M_with_exp_decay(S::Vector{Float64})
    if isempty(S)
        return 0
    end
    
    logS = log.(S)
    
    knee = _find_knee(logS)
    
    if knee < 2
        return knee
    end
    
    S_truncated = S[1:knee]
    
    dlogS = abs.(diff(log.(S_truncated)))
    
    if isempty(dlogS)
        return 1
    end
    
    threshold = maximum(dlogS) / 3.0
    indices = findall(dlogS .> threshold)
    
    if isempty(indices)
        return 1
    end
    
    M = last(indices)
    return M
end

function _find_roots_from_companion_matrix(a::AbstractVector{<:Number})
    C = zeros(ComplexF64, length(a), length(a))
    for k in 1:length(a)-1
        C[k+1, k] = 1.0
    end
    C[:, end] .= -a
    return eigvals(C)
end