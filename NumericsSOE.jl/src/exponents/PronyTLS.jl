export PronyTLS, PronyTLSOutput

struct PronyTLSOutput{T_z, T_S} <: AbstractExponentsOutput
    z::T_z
    S::T_S
end

struct PronyTLS <: AbstractExponentMethod
    opts::CommonOptions
    M::Union{Int, Nothing}
    tolerance::Union{Float64, Nothing}
    L_factor::Float64
end

function PronyTLS(; M=nothing, tolerance=nothing, L_factor=0.4, verbose::Bool=false)
    if M isa Int && M <= 0
        throw(ArgumentError("Model order M must be a positive integer."))
    end
    if tolerance isa Float64 && tolerance <= 0
        throw(ArgumentError("Tolerance must be a positive number."))
    end
    opts = CommonOptions(verbose=verbose)
    return PronyTLS(opts, M, tolerance, L_factor)
end

"""
    _find_discrete_eigenvalues(f, t, method::PronyTLS)

Kernel function for the Prony-TLS method.

This follows the logic from `prony_approx.py`:
1. Build the square (N+1)x(N+1) Hankel matrix `H`.
2. Perform SVD: `U, S, V = svd(H)`.
3. Estimate model order `M_eff`.
4. Get the TLS polynomial coefficients `v` from `V[:, M_eff + 1]`.
5. Find all roots `z_all` of the polynomial `P(z) = sum(v[k] * z^(k-1))`.
6. Filter roots: `z_filtered = z_all[abs.(z_all) .< method.cutoff]`.
7. Find amplitudes `omega` for `z_filtered` by solving `A * omega = f` via `pinv(A)`.
8. Sort `z_filtered` by `abs.(omega)` in descending order.
9. Return the top `M_eff` poles.
"""
function _find_discrete_eigenvalues(f::AbstractArray{<:Number}, t::Vector{<:Real}, method::PronyTLS)
    # This function ignores `t` as Prony assumes uniform sampling.
    f_reshaped = _prepare_signal_for_hankel(f)
    
    K = length(f_reshaped)
    if !isodd(K)
        throw(ArgumentError("Signal length must be odd for Prony-TLS. Received length $K."))
    end
    N = (K - 1) ÷ 2

    # Build the (N+1) x (N+1) Hankel matrix H
    # H[i, j] = f[i+j-1]
    H = zeros(eltype(f_reshaped), N + 1, N + 1)
    for row in 1:(N + 1)
        # f_reshaped is 1-indexed. Python h_k[row : (row + N + 1)]
        # maps to f_reshaped[row : (row + N)]
        H[row, :] = f_reshaped[row:(row + N)]
    end

    # Get SVD
    U, S, V = svd(H)

    # Estimate model order
    M_eff = _estimate_model_order(method.M, method.tolerance, method.opts.verbose, S)

    # Check that M_eff is valid
    if M_eff >= length(S)
        throw(ArgumentError("Estimated model order M = $M_eff is not less than the number of singular values $(length(S)). Cannot select TLS vector."))
    end

    # Get the TLS solution vector (coefficients of the polynomial)
    # This is the right singular vector corresponding to the first "noise" singular value
    v = V[:, M_eff + 1]

    # The polynomial is P(z) = v[1] + v[2]*z + ... + v[N+1]*z^N.
    # To use the companion matrix method, we need to find the roots of
    # z^N + a_1 z^{N-1} + ... + a_N = 0, where a = [a_N, ..., a_1].
    # The coefficients are a_k = v[N+1-k]/v[N+1].
    # The vector for the function is [a_N, ..., a_1], which is
    # [v[1]/v[N+1], ..., v[N]/v[N+1]].
    a = v[1:end-1] ./ v[end]
    z_all = _find_roots_from_companion_matrix(a)

    # Filter roots based on the cutoff (e.g., poles inside the unit circle)
    z_filtered = z_all[abs.(z_all) .< method.cutoff]

    if isempty(z_filtered)
        if method.opts.verbose
            println("Warning: No poles found within cutoff radius $(method.cutoff). Returning empty result.")
        end
        return PronyTLSOutput(eltype(z_all)[], S)
    end

    # --- Find amplitudes to select the most significant poles ---
    
    # Build the Vandermonde matrix A (K x N_filtered_poles)
    # A[k, j] = z_filtered[j]^(k-1) for k in 1:K
    # We use (0:(K-1)) for the exponents, matching the Python code's `self.gamma ** i`
    A = (z_filtered') .^ (0:(K-1))

    # Solve the least-squares problem for amplitudes: A * omega = f
    # Use pinv for stability, matching np.linalg.pinv
    omega = pinv(A) * f_reshaped

    # Sort poles by the magnitude of their amplitudes (descending)
    sort_indices = sortperm(abs.(omega), rev=true)

    # Select the top M_eff poles
    num_to_keep = min(M_eff, length(z_filtered))
    final_indices = sort_indices[1:num_to_keep]
    z = z_filtered[final_indices]

    return PronyTLSOutput(z, S)
end