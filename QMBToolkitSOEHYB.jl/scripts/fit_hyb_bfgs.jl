#
# Standalone script: fit a hybridization function Δ(iω_n) on a Matsubara grid
# with a sum-of-poles ansatz
#
#     Δ(iω_n) ≈ Σ_{ℓ=1}^{Np}  V_ℓ V_ℓ^T  / (i ω_n - λ_ℓ)
#
# where λ_ℓ ∈ ℝ and V_ℓ ∈ ℝ^{n_flavor} is a REAL vector, so V_ℓ V_ℓ^T is a
# rank-1 real symmetric PSD matrix of size n_flavor × n_flavor. (This assumes
# a time-reversal-symmetric / real-hopping problem.) The cost function
# (Eq. 9 of the reference) is
#
#     J({λ_ℓ, V_ℓ}) = (1/N_ω) * sqrt( Σ_n || Δ(iω_n) - Σ_ℓ V_ℓ V_ℓ^T /(iω_n - λ_ℓ) ||_F^2 )
#
# and we minimize it with BFGS from Optim.jl.
#
# Parameters are packed as a real vector
#     x = [ λ_1, ..., λ_{Np},
#           V_1[1], ..., V_1[n_flavor],
#           V_2[1], ..., V_2[n_flavor],
#           ... ]
#
# Running this file executes an artificial test that generates several
# synthetic Δ(iω_n) from known (λ_true, V_true), fits each with BFGS starting
# from a random initial guess, and checks that the fitted Δ reproduces the
# target to high precision.
#

using LinearAlgebra
using Printf
using Random
using Optim
using ForwardDiff

# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

n_params(Np::Int, n_flavor::Int) = Np + Np * n_flavor

function pack_params(λ::AbstractVector{<:Real}, V::AbstractMatrix{<:Real})
    Np = length(λ)
    n_flavor = size(V, 2)
    @assert size(V, 1) == Np
    x = Vector{Float64}(undef, n_params(Np, n_flavor))
    x[1:Np] .= λ
    off = Np
    for ℓ in 1:Np
        for i in 1:n_flavor
            x[off + i] = V[ℓ, i]
        end
        off += n_flavor
    end
    return x
end

function unpack_params(x::AbstractVector{T}, Np::Int, n_flavor::Int) where {T<:Real}
    λ = x[1:Np]
    V = Matrix{T}(undef, Np, n_flavor)
    off = Np
    for ℓ in 1:Np
        for i in 1:n_flavor
            V[ℓ, i] = x[off + i]
        end
        off += n_flavor
    end
    return λ, V
end

# ---------------------------------------------------------------------------
# Reference model evaluation (for generating synthetic data and reporting).
# ---------------------------------------------------------------------------

"""
    model_delta(λ, V, ω)

Evaluate Δ_model(iω_n) = Σ_ℓ V_ℓ V_ℓ^T / (iω_n - λ_ℓ) on the grid `iω = im .* ω`
with real λ and real V. Returns a complex array of shape `(N_ω, n_flavor, n_flavor)`.
"""
function model_delta(λ::AbstractVector{<:Real}, V::AbstractMatrix{<:Real},
                     ω::AbstractVector{<:Real})
    Np = length(λ)
    n_flavor = size(V, 2)
    N_ω = length(ω)
    Δ = zeros(ComplexF64, N_ω, n_flavor, n_flavor)
    for n in 1:N_ω
        iωn = im * ω[n]
        for ℓ in 1:Np
            factor = 1.0 / (iωn - λ[ℓ])
            for i in 1:n_flavor, j in 1:n_flavor
                Δ[n, i, j] += V[ℓ, i] * V[ℓ, j] * factor
            end
        end
    end
    return Δ
end

# ---------------------------------------------------------------------------
# Cost function (real-valued, ForwardDiff-friendly).
#
# We minimize the sum of squared Frobenius norms
#     S(x) = Σ_n || Δ_target(iω_n) - Δ_model(iω_n; x) ||_F^2
# which has the same minimizer as J(x) = sqrt(S(x))/N_ω but avoids the sqrt
# singularity at the optimum. J is reported at the end.
# ---------------------------------------------------------------------------

function cost_sq(x::AbstractVector{T}, Np::Int, n_flavor::Int,
                 ω::AbstractVector{Float64},
                 Δ_re::AbstractArray{Float64,3},
                 Δ_im::AbstractArray{Float64,3}) where {T<:Real}
    N_ω = length(ω)
    total = zero(T)
    @inbounds for n in 1:N_ω
        ωn = ω[n]
        for i in 1:n_flavor, j in 1:n_flavor
            mre = zero(T)
            mim = zero(T)
            for ℓ in 1:Np
                λℓ = x[ℓ]
                # 1/(iω_n - λ_ℓ) = -(λ_ℓ + i ω_n)/(ω_n^2 + λ_ℓ^2)
                denom = ωn * ωn + λℓ * λℓ
                inv_re = -λℓ / denom
                inv_im = -ωn / denom
                off_V = Np + (ℓ - 1) * n_flavor
                Vi = x[off_V + i]
                Vj = x[off_V + j]
                # V_i * V_j (both real) times (inv_re + i inv_im)
                num = Vi * Vj
                mre += num * inv_re
                mim += num * inv_im
            end
            dre = mre - Δ_re[n, i, j]
            dim = mim - Δ_im[n, i, j]
            total += dre * dre + dim * dim
        end
    end
    return total
end

J_from_cost_sq(cs::Real, N_ω::Int) = sqrt(max(cs, 0.0)) / N_ω

# ---------------------------------------------------------------------------
# BFGS driver
# ---------------------------------------------------------------------------

"""
    fit_hyb_bfgs(Δ_target, ω, Np; ...)

Fit the complex array `Δ_target` of shape `(N_ω, n_flavor, n_flavor)` sampled at
the fermionic Matsubara frequencies `iω_n = im*ω[n]` with `Np` poles using BFGS.

Returns `(λ, V, J, result)`.
"""
function fit_hyb_bfgs(
    Δ_target::AbstractArray{ComplexF64,3},
    ω::AbstractVector{Float64},
    Np::Int;
    λ0::Union{Nothing,AbstractVector{<:Real}} = nothing,
    V0::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    rng_seed::Int = 0,
    g_tol::Float64 = 1e-14,
    iterations::Int = 2000,
    show_trace::Bool = false,
)
    N_ω, n_flavor, n_flavor2 = size(Δ_target)
    @assert n_flavor == n_flavor2 "Δ_target must be square in flavor dimensions"
    @assert length(ω) == N_ω "ω must have length N_ω = size(Δ_target, 1)"

    rng = MersenneTwister(rng_seed)
    λ_init = λ0 === nothing ? collect(range(-2.0, 2.0; length = Np)) .+ 0.05 .* randn(rng, Np) : collect(Float64, λ0)
    V_init = V0 === nothing ? 0.1 .* randn(rng, Float64, Np, n_flavor) : Float64.(V0)

    x0 = pack_params(λ_init, V_init)
    ω_f = collect(Float64, ω)
    Δ_re = Array{Float64,3}(real.(Δ_target))
    Δ_im = Array{Float64,3}(imag.(Δ_target))

    f = x -> cost_sq(x, Np, n_flavor, ω_f, Δ_re, Δ_im)
    cfg = ForwardDiff.GradientConfig(f, x0)
    g! = (G, x) -> ForwardDiff.gradient!(G, f, x, cfg)

    opts = Optim.Options(
        g_tol = g_tol,
        iterations = iterations,
        show_trace = show_trace,
        show_every = 50,
    )

    result = optimize(f, g!, x0, BFGS(), opts)

    x_opt = Optim.minimizer(result)
    λ_opt, V_opt = unpack_params(x_opt, Np, n_flavor)
    J_opt = J_from_cost_sq(Optim.minimum(result), N_ω)
    return λ_opt, V_opt, J_opt, result
end

# ---------------------------------------------------------------------------
# Matsubara grid helper
# ---------------------------------------------------------------------------

"""
    fermionic_matsubara(beta, N)

Return the first `N` positive fermionic Matsubara frequencies
`ω_n = (2n+1) π / β` as real numbers (multiply by `im` for `iω_n`).
"""
fermionic_matsubara(beta::Float64, N::Int) = Float64[(2 * n + 1) * pi / beta for n in 0:N-1]

# ---------------------------------------------------------------------------
# Artificial tests
# ---------------------------------------------------------------------------

struct TestCase
    name::String
    beta::Float64
    λ_true::Vector{Float64}
    V_true::Matrix{Float64}
end

function build_test_cases()
    cases = TestCase[]

    # --- Case 1: single flavor, 2 well-separated poles, β = 5 -------------
    push!(cases, TestCase(
        "1 flavor,  2 poles,  β = 5",
        5.0,
        [-1.2, 1.4],
        reshape(Float64[1.0, 0.8], 2, 1),
    ))

    # --- Case 2: single flavor, 3 poles, β = 10 ---------------------------
    push!(cases, TestCase(
        "1 flavor,  3 poles,  β = 10",
        10.0,
        [-1.5, 0.3, 2.1],
        reshape(Float64[1.0, 0.7, 0.5], 3, 1),
    ))

    # --- Case 3: single flavor, 3 poles, low-T β = 50 ---------------------
    push!(cases, TestCase(
        "1 flavor,  3 poles,  β = 50",
        50.0,
        [-0.8, 0.1, 1.6],
        reshape(Float64[0.9, 0.4, 0.6], 3, 1),
    ))

    # --- Case 4: 2 flavors, 3 poles, β = 10 -------------------------------
    push!(cases, TestCase(
        "2 flavors, 3 poles,  β = 10",
        10.0,
        [-1.5, 0.3, 2.1],
        Float64[
            1.0   0.4;
            0.2   0.8;
            0.5   0.1
        ],
    ))

    # --- Case 5: 2 flavors, 3 poles, high-T β = 2 -------------------------
    push!(cases, TestCase(
        "2 flavors, 3 poles,  β = 2  (high T)",
        2.0,
        [-1.8, 0.0, 1.9],
        Float64[
             0.8   0.3;
            -0.4   0.6;
             0.5  -0.2
        ],
    ))

    # --- Case 6: 3 flavors, 4 poles, β = 10 -------------------------------
    push!(cases, TestCase(
        "3 flavors, 4 poles,  β = 10",
        10.0,
        [-2.3, -0.7, 0.5, 1.9],
        Float64[
             1.2  -0.2   0.5;
             0.4   0.9   0.1;
            -0.3   0.2   0.7;
             0.6   0.1  -0.4
        ],
    ))

    # --- Case 7: 2 flavors, asymmetric poles (all positive), β = 20 ------
    push!(cases, TestCase(
        "2 flavors, 3 poles (all > 0), β = 20",
        20.0,
        [0.2, 0.9, 2.4],
        Float64[
             0.7   0.5;
             0.3  -0.6;
            -0.4   0.2
        ],
    ))

    # --- Case 8: 6 flavors, 6 poles, low-T β = 100 -----------------------
    push!(cases, TestCase(
        "6 flavors, 6 poles,  β = 100",
        100.0,
        [-2.5, -1.4, -0.3, 0.4, 1.2, 2.3],
        Float64[
             0.9  -0.2   0.3   0.1  -0.4   0.2;
             0.4   0.7  -0.1   0.3   0.2  -0.3;
            -0.3   0.5   0.6  -0.2   0.1   0.4;
             0.2  -0.1   0.4   0.8  -0.3   0.1;
             0.5   0.3  -0.2   0.1   0.7  -0.4;
            -0.1   0.4   0.3  -0.5   0.2   0.6
        ],
    ))

    return cases
end

function run_one_test(tc::TestCase; N_ω::Int = 200, rng_seed::Int = 42,
                       err_tol::Float64 = 1e-6)
    Np_true = length(tc.λ_true)
    n_flavor = size(tc.V_true, 2)
    ω = fermionic_matsubara(tc.beta, N_ω)
    Δ_target = model_delta(tc.λ_true, tc.V_true, ω)

    println("\n=============================================================")
    println(" Test: ", tc.name)
    println("=============================================================")
    @printf("  β = %.2f, N_ω = %d, Np_true = %d, n_flavor = %d, max|Δ_target| = %.3e\n",
            tc.beta, N_ω, Np_true, n_flavor, maximum(abs.(Δ_target)))

    # --- Fit with the correct number of poles, timed ---
    t_start = time_ns()
    λ_fit, V_fit, J_fit, res = fit_hyb_bfgs(
        Δ_target, ω, Np_true;
        rng_seed = rng_seed,
        show_trace = false,
    )
    elapsed = (time_ns() - t_start) / 1e9  # seconds

    Δ_fit = model_delta(λ_fit, V_fit, ω)
    max_err = maximum(abs.(Δ_fit .- Δ_target))

    @printf("  BFGS iterations    : %d\n", Optim.iterations(res))
    @printf("  elapsed time       : %.3f s\n", elapsed)
    @printf("  J (fit)            : %.6e\n", J_fit)
    @printf("  max|Δ_fit-Δ_target|: %.6e\n", max_err)
    @printf("  λ_true (sorted)    : %s\n", string(round.(sort(tc.λ_true); digits = 6)))
    @printf("  λ_fit  (sorted)    : %s\n", string(round.(sort(λ_fit); digits = 6)))

    passed = max_err < err_tol
    println("  status             : ", passed ? "PASSED" : "FAILED")
    return passed, max_err, J_fit, elapsed
end

function run_artificial_tests()
    println("=== Artificial tests: BFGS fit of Δ(iω_n) with sum-of-poles ansatz ===")
    cases = build_test_cases()

    # --- Warm-up run to exclude JIT compilation from timings -------------
    println("\n(warm-up run to trigger JIT compilation, timings excluded)")
    run_one_test(cases[1]; rng_seed = 1)

    results = Bool[]
    times   = Float64[]
    for (k, tc) in enumerate(cases)
        passed, _, _, elapsed = run_one_test(tc; rng_seed = 42 + k)
        push!(results, passed)
        push!(times, elapsed)
    end

    println("\n=== Summary ===")
    for (tc, ok, t) in zip(cases, results, times)
        @printf("  [%s]  %-40s  %8.3f s\n", ok ? "PASS" : "FAIL", tc.name, t)
    end
    n_pass = count(results)
    @printf("%d / %d tests passed   (total wall time: %.3f s)\n",
            n_pass, length(results), sum(times))
    return n_pass == length(results)
end

run_artificial_tests()
