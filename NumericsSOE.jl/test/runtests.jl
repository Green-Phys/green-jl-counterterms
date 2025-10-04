using NumericsESPRIT
using Test
using Random

@testset "NumericsESPRIT.jl" begin
    @testset "esprit noiseless" begin

        # Create a test signal
        N = 100
        a = 2.0
        b = 3.0
        dt = (b - a) / (N - 1)
        t = collect(a:dt:b)
        s1 = 0.1 + 0.05im
        s2 = 0.2 + 0.03im
        R1 = 1.0 + 0.5im
        R2 = 0.5 + 0.2im
        true_s = [s1, s2]
        true_R = [R1, R2]
        f = R1 * exp.(s1 * t) + R2 * exp.(s2 * t)

        # ESPRIT parameters
        M = 2

        # Estimate results
        s_est, R_est = esprit(f, t, M)
        f_rec = reconstruct_signal(s_est, R_est, t)

        # Sort results for comparison
        perm_true = sortperm(real.(true_s))
        perm_est = sortperm(real.(s_est))

        sorted_true_s = true_s[perm_true]
        sorted_est_s = s_est[perm_est]

        sorted_true_R = true_R[perm_true]
        sorted_est_R = R_est[perm_est]

        @test sorted_est_s ≈ sorted_true_s
        @test sorted_est_R ≈ sorted_true_R
        @test f_rec ≈ f

    end
end