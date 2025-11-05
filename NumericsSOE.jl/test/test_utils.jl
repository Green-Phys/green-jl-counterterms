# This file contains shared functions for generating test data.
# It assumes that the main `runtests.jl` has already done `using NumericsSOE`.

using Random
using Test

function get_noiseless_data()
    N = 100
    a = 2.0
    b = 3.0
    t = collect(range(a, stop=b, length=N))
    s1 = 0.1 + 0.05im
    s2 = 0.2 + 0.03im
    R1 = 1.0 + 0.5im
    R2 = 0.5 + 0.2im
    true_s = [s1, s2]
    true_R = [R1, R2]
    f = R1 * exp.(s1 * t) + R2 * exp.(s2 * t)
    return (; f, t, true_s, true_R)
end

function get_tensor_data()
    N = 100
    t = collect(range(0.0, stop=10.0, length=N))
    dims = (2, 3, 4)
    true_s = [0.1 + 2.0im, -0.05 + 5.0im]
    M = length(true_s)
    Random.seed!(1234)
    true_R = rand(ComplexF64, M, dims...)
    f = reconstruct_signal(true_s, true_R, t)
    return (; f, t, true_s, true_R, dims)
end

function get_noisy_data()
    N = 100
    t = collect(range(0.0, stop=5.0, length=N))
    true_s = [-0.1 + 4.0im, -0.2 + 8.0im, -0.3 + 12.0im]
    true_R = [1.0 + 0.0im, 0.5 + 0.0im, 0.2 + 0.0im]
    f_clean = reconstruct_signal(true_s, true_R, t)
    Random.seed!(4321)
    noise_level = 0.01
    noise = noise_level * (randn(N) + 1im * randn(N)) / sqrt(2)
    f_noisy = f_clean + noise
    return (; f_noisy, t, true_s, true_R)
end

function assert_results_approx_equal(s_est, R_est, true_s, true_R; by=real, atol=1e-8, rtol=isapprox(atol, 0) ? 1e-8 : 0)
    # Sort results for comparison
    perm_true = sortperm(by.(true_s))
    perm_est = sortperm(by.(s_est))

    sorted_true_s = true_s[perm_true]
    sorted_est_s = s_est[perm_est]

    # Handle both vector and tensor-valued R
    if ndims(true_R) == 1
        sorted_true_R = true_R[perm_true]
        sorted_est_R = R_est[perm_est]
    else
        # For tensors, permute along the first dimension
        sorted_true_R = true_R[perm_true, ntuple(_ -> Colon(), ndims(true_R) - 1)...]
        sorted_est_R = R_est[perm_est, ntuple(_ -> Colon(), ndims(R_est) - 1)...]
    end

    @test sorted_est_s ≈ sorted_true_s atol=atol rtol=rtol
    @test sorted_est_R ≈ sorted_true_R atol=atol rtol=rtol
end
