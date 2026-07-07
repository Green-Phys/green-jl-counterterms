using NumericsSOE
using Test
using Random

@testset "PronyTLS Constructor Validation" begin
    @test_throws ArgumentError PronyTLS(L_factor = 0.1)
    @test_throws ArgumentError PronyTLS(L_factor = 0.6)
    @test_throws ArgumentError PronyTLS(M = -1)
    @test_throws ArgumentError PronyTLS(tolerance = 0.0)
    @test_throws ArgumentError PronyTLS(tolerance = -1e-9)
end

@testset "PronyTLS noiseless" begin
    data = get_noiseless_data()
    # The TLS method requires an odd number of points for its default Hankel matrix construction
    f_odd = data.f[1:end-1]
    t_odd = data.t[1:end-1]

    solution = solve(f_odd, t_odd, PronyTLS(M=length(data.true_s)), LeastSquares())
    s_est = solution.s
    R_est = solution.R
    f_rec = reconstruct_signal(s_est, R_est, t_odd)

    assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R, atol=1e-6)
    @test f_rec ≈ f_odd atol=1e-6
end

@testset "PronyTLS noiseless tensor" begin
    data = get_tensor_data()
    # The TLS method requires an odd number of points
    f_odd = data.f[1:end-1, :, :, :]
    t_odd = data.t[1:end-1]

    solution = solve(f_odd, t_odd, PronyTLS(M=length(data.true_s)), LeastSquares())
    s_est = solution.s
    R_est = solution.R
    f_rec = reconstruct_signal(s_est, R_est, t_odd)

    assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R, by=imag, atol=1e-6)
    @test f_rec ≈ f_odd atol=1e-6
end

@testset "PronyTLS noiseless auto M threshold" begin
    data = get_noiseless_data()
    f_odd = data.f[1:end-1]
    t_odd = data.t[1:end-1]

    solution = solve(f_odd, t_odd, PronyTLS(tolerance=1e-12), LeastSquares())
    s_est = solution.s
    R_est = solution.R
    f_rec = reconstruct_signal(s_est, R_est, t_odd)
    M_est = length(s_est)

    @test M_est == 2
    assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R, atol=1e-6)
    @test f_rec ≈ f_odd atol=1e-6
end

@testset "PronyTLS noisy auto M exp_decay" begin
    data = get_noisy_data()
    f_odd = data.f_noisy[1:end-1]
    t_odd = data.t[1:end-1]
    M_true = length(data.true_s)

    solution = solve(f_odd, t_odd, PronyTLS(), LeastSquares())
    s_est = solution.s
    R_est = solution.R
    
    M_est = length(s_est)
    @test M_est == M_true

    if M_est == M_true
        assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R, by=imag, atol=0.5)
    end
end
