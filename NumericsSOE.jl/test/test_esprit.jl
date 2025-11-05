using NumericsSOE
using Test
using Random

@testset "Constructor Validation" begin
    @test_throws ArgumentError ESPRIT(L_factor = 0.1)
    @test_throws ArgumentError ESPRIT(L_factor = 0.6)
    @test_throws ArgumentError ESPRIT(M = -1)
    @test_throws ArgumentError ESPRIT(tolerance = 0.0)
    @test_throws ArgumentError ESPRIT(tolerance = -1e-9)
end

@testset "esprit noiseless" begin
    data = get_noiseless_data()
    s_est, R_est, _, _ = solve(data.f, data.t, ESPRIT(), LeastSquares())
    f_rec = reconstruct_signal(s_est, R_est, data.t)

    assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R)
    @test f_rec ≈ data.f
end

@testset "esprit noiseless tensor" begin
    data = get_tensor_data()
    s_est, R_est, _, _ = solve(data.f, data.t, ESPRIT(M=length(data.true_s)), LeastSquares())
    f_rec = reconstruct_signal(s_est, R_est, data.t)

    assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R, by=imag)
    @test f_rec ≈ data.f
end

@testset "esprit noiseless auto M threshold" begin
    data = get_noiseless_data()
    s_est, R_est, _, M_est = solve(data.f, data.t, ESPRIT(tolerance=1e-9), LeastSquares())
    f_rec = reconstruct_signal(s_est, R_est, data.t)

    @test M_est == 2
    assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R)
    @test f_rec ≈ data.f
end

@testset "esprit noisy auto M exp_decay" begin
    data = get_noisy_data()
    M_true = length(data.true_s)

    s_est, R_est, _, M_est = solve(data.f_noisy, data.t, ESPRIT(), LeastSquares())

    @test M_est == M_true

    if M_est == M_true
        assert_results_approx_equal(s_est, R_est, data.true_s, data.true_R, by=imag, atol=0.05)
    end
end