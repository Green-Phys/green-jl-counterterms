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

    @testset "esprit noiseless tensor" begin
        # Signal parameters
        N = 100
        t = collect(range(0.0, stop=10.0, length=N))
        dims = (2, 3, 4)

        # True parameters
        true_s = [0.1 + 2.0im, -0.05 + 5.0im]
        M = length(true_s)
        
        # Generate random amplitudes
        Random.seed!(1234)
        true_R = rand(ComplexF64, M, dims...)

        # Generate signal using reconstruct_signal
        f = reconstruct_signal(true_s, true_R, t)

        # Estimate parameters
        s_est, R_est = esprit(f, t, M)

        # Reconstruct signal from estimated parameters
        f_rec = reconstruct_signal(s_est, R_est, t)

        # Sort for comparison
        perm_true = sortperm(imag.(true_s))
        perm_est = sortperm(imag.(s_est))

        sorted_true_s = true_s[perm_true]
        sorted_est_s = s_est[perm_est]

        # Permute amplitudes (first dimension) according to pole sort order
        sorted_true_R = true_R[perm_true, :, :, :]
        sorted_est_R = R_est[perm_est, :, :, :]

        @test sorted_est_s ≈ sorted_true_s
        @test sorted_est_R ≈ sorted_true_R
        @test f_rec ≈ f
    end

    @testset "esprit noiseless auto M" begin

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

        # Estimate results with M=nothing
        s_est, R_est = esprit(f, t)
        f_rec = reconstruct_signal(s_est, R_est, t)

        # Sort results for comparison
        perm_true = sortperm(real.(true_s))
        perm_est = sortperm(real.(s_est))

        sorted_true_s = true_s[perm_true]
        sorted_est_s = s_est[perm_est]

        sorted_true_R = true_R[perm_true]
        sorted_est_R = R_est[perm_est]

        @test length(s_est) == 2
        @test sorted_est_s ≈ sorted_true_s
        @test sorted_est_R ≈ sorted_true_R
        @test f_rec ≈ f

    end

    @testset "esprit noisy" begin
        # Signal parameters
        N = 100
        t = collect(range(0.0, stop=5.0, length=N))
        true_s = [-0.1 + 4.0im, -0.2 + 8.0im]
        true_R = [1.0 + 0.0im, 0.5 + 0.0im]
        M_true = length(true_s)
        
        f_clean = reconstruct_signal(true_s, true_R, t)

        # Add noise
        Random.seed!(4321)
        noise_level = 0.01
        noise = noise_level * (randn(N) + 1im * randn(N)) / sqrt(2)
        f_noisy = f_clean + noise

        # ESPRIT parameters
        # M is automatic. The tolerance needs to be chosen to separate
        # the signal singular values from the noise singular values.
        # A larger tolerance makes the estimation more selective.
        M=2

        # Estimate results
        s_est, R_est = esprit(f_noisy, t, M)

        # Check model order
        @test length(s_est) == M_true

        # Sort results for comparison
        perm_true = sortperm(imag.(true_s))
        perm_est = sortperm(imag.(s_est))

        sorted_true_s = true_s[perm_true]
        sorted_est_s = s_est[perm_est]

        sorted_true_R = true_R[perm_true]
        sorted_est_R = R_est[perm_est]

        # Compare with a larger tolerance for the noisy case
        @test sorted_est_s ≈ sorted_true_s atol=1e-2
        @test sorted_est_R ≈ sorted_true_R atol=1e-1
    end
end