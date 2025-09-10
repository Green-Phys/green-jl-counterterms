using QMBToolkitED
using Test
using LinearAlgebra
using SparseArrays

@testset "Fermionic Operators" begin
    nflavours = 6
    c = get_annihilators(nflavours)
    c_dag = [op' for op in c]
    nfock = 1 << nflavours

    # Test {c_i, c_j} = 0 for i != j
    @testset "Anticommutator {c_i, c_j}" begin
        for i in 1:nflavours
            for j in (i+1):nflavours
                @test Matrix(c[i] * c[j] + c[j] * c[i]) ≈ spzeros(nfock, nfock) atol=1e-9
            end
        end
    end

    # Test {c_i, c_j_dag} = delta_ij * I
    @testset "Anticommutator {c_i, c_j_dag}" begin
        for i in 1:nflavours
            for j in 1:nflavours
                result = Matrix(c[i] * c_dag[j] + c_dag[j] * c[i])
                if i == j
                    @test result ≈ Matrix(I, nfock, nfock) atol=1e-9
                else
                    @test result ≈ spzeros(nfock, nfock) atol=1e-9
                end
            end
        end
    end
end


@testset "Anderson Impurity Model" begin
    # impurity site + 2 bath sites = 6 flavours
    # U=1.0, mu=0.5, epsk=[-0.5, 0.5], vk=[1.0, 1.0]
    hopping_str = """
    0 0 -0.5 0.0
    1 1 -0.5 0.0
    2 2 -0.5 0.0
    3 3 -0.5 0.0
    4 4 0.5 0.0
    5 5 0.5 0.0
    2 0 1.0 0.0
    0 2 1.0 0.0
    3 1 1.0 0.0
    1 3 1.0 0.0
    4 0 1.0 0.0
    0 4 1.0 0.0
    5 1 1.0 0.0
    1 5 1.0 0.0
    """
    # U_{0110} = 1.0, U_{1001} = 1.0 for U=1.0, since H_int = 0.5 * U_ijkl c_i'c_j'c_k c_l
    coulomb_str = """
    1
    0 0 1 1 0 1.0 0.0
    1 1 0 0 1 1.0 0.0
    """

    # 1. Build Hamiltonian
    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    nflavours = 6
    nfock = 1 << nflavours
    @test size(H) == (nfock, nfock)
    @test ishermitian(H)

    # 2. Solve Hamiltonian and check partition function
    evals, evecs = solve_hamiltonian(H)
    @test length(evals) == nfock
    
    # Partition function
    beta = 10.0
    Z = get_Z(evals, beta, shift_energy=true)
    
    # ED value from other implementations
    @test Z ≈ 3.87936702286 atol=1e-6
end

@testset "Non-interacting Model" begin
    # Non-interacting single-site model: H = ϵ c^† c
    ϵ = 0.5
    hopping_str = "0 0 $(ϵ) 0.0"
    coulomb_str = "0"

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    evals, evecs = solve_hamiltonian(H)

    beta = 10.0

    @testset "Matsubara" begin
        n_matsubara = 100
        iwn, giwn = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)

        # Analytical result: G(iω_n) = 1 / (iω_n - ϵ)
        giwn_analytical = 1 ./ (iwn .- ϵ)

        @test giwn ≈ giwn_analytical atol=1e-9
    end

    @testset "Imaginary-Time" begin
        n_tau = 100
        tau_grid, gtau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1)

        # Analytical result: G(τ) = -exp(-ϵτ) / (1 + exp(-βϵ))
        Z_analytical = 1 + exp(-beta * ϵ)
        gtau_analytical = -exp.(-ϵ .* tau_grid) ./ Z_analytical

        @test gtau ≈ gtau_analytical atol=1e-9
    end
end

@testset "Hubbard Atom" begin
    U = 2.0
    beta = 10.0

    # Hamiltonian H = U * n_up * n_down - (U/2) * (n_up + n_down)
    # This corresponds to a particle-hole symmetric Hubbard atom at half-filling.
    hopping_str = """
    0 0 $(-U/2) 0.0
    1 1 $(-U/2) 0.0
    """
    coulomb_str = """
    2
    0 0 1 1 0 $(U) 0.0
    1 1 0 0 1 $(U) 0.0
    """

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    evals, evecs = solve_hamiltonian(H)

    @testset "Matsubara" begin
        n_matsubara = 100
        # Test for c_up
        iwn, giwn = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)

        # Analytical result: G(iω_n) = 0.5 * (1/(iω_n + U/2) + 1/(iω_n - U/2))
        giwn_analytical = 0.5 .* (1 ./ (iwn .+ U/2) .+ 1 ./ (iwn .- U/2))

        @test giwn ≈ giwn_analytical atol=1e-9

        # Off-diagonal terms should be zero
        _, giwn_offdiag = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 2)
        @test all(isapprox.(giwn_offdiag, 0.0, atol=1e-9))
    end

    @testset "Imaginary-Time" begin
        n_tau = 100
        # Test for c_up
        tau_grid, gtau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1)

        # Analytical result derived from the Lehmann representation, consistent with the code's calculation.
        Z_analytical = 2 * (1 + exp(-beta * U/2))
        gtau_analytical = -(exp.(-(beta .- tau_grid) .* (U/2)) .+ exp.(-tau_grid .* (U/2))) ./ Z_analytical

        @test gtau ≈ gtau_analytical atol=1e-9

        # Off-diagonal terms should be zero
        _, gtau_offdiag = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 2)
        @test all(isapprox.(gtau_offdiag, 0.0, atol=1e-9))
    end
end

@testset "Hubbard Dimer" begin
    @testset "Non-degenerate Ground State" begin
        t = 1.0
        U = 2.0
        beta = 20.0 # Low temperature for the analytical approximation to be valid
        n_matsubara = 100

        # Flavors: 0 (site 1, up), 1 (site 1, down), 2 (site 2, up), 3 (site 2, down)
        # The analytical formula assumes mu=U/2, so we add the on-site energy term -U/2.
        hopping_str = """
        0 0 $(-U/2) 0.0
        1 1 $(-U/2) 0.0
        2 2 $(-U/2) 0.0
        3 3 $(-U/2) 0.0
        0 2 $(-t) 0.0
        2 0 $(-t) 0.0
        1 3 $(-t) 0.0
        3 1 $(-t) 0.0
        """
        coulomb_str = """
        4
        0 0 1 1 0 $(U) 0.0
        1 1 0 0 1 $(U) 0.0
        2 2 3 3 2 $(U) 0.0
        3 3 2 2 3 $(U) 0.0
        """

        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
        evals, evecs = solve_hamiltonian(H)

        # Calculate numerical Green's function
        # G_11 (site 1 up, site 1 up)
        iwn, g11_numeric = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)
        # G_12 (site 1 up, site 2 up)
        _, g12_numeric = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 3)

        # Analytical formula (approximation for large t, low T)
        c = sqrt(16*t^2 + U^2)
        c_minus_U = c - U
        # Note: formula uses a^2, so we calculate that directly
        a_sq = (32*t^2 + 2*c_minus_U^2) / c_minus_U^2
        
        term1_factor = (1 + 4*t / c_minus_U)^2
        term2_factor = (1 - 4*t / c_minus_U)^2

        # Assuming (-1)^ij in the formula corresponds to (-1)^(i+j) for sites i,j in {1,2}
        # For G_11, i=j=1, so the sign-changing factor is 1
        g11_analytical = (1 / (2 * a_sq)) .* (
            (term1_factor ./ (iwn .+ t .- c/2)) .+
            (term2_factor ./ (iwn .- t .- c/2)) .+
            (term1_factor ./ (iwn .- t .+ c/2)) .+
            (term2_factor ./ (iwn .+ t .+ c/2))
        )

        # For G_12, i=1, j=2, so the sign-changing factor is -1
        g12_analytical = (1 / (2 * a_sq)) .* (
            (-term1_factor ./ (iwn .+ t .- c/2)) .+
            (term2_factor ./ (iwn .- t .- c/2)) .+
            (term1_factor ./ (iwn .- t .+ c/2)) .+
            (-term2_factor ./ (iwn .+ t .+ c/2))
        )
        
        # Using a larger tolerance because the analytical formula is an approximation
        @test g11_numeric ≈ g11_analytical atol=1e-6
        @test g12_numeric ≈ g12_analytical atol=1e-6
    end

    @testset "Almost Degenerate Ground State (t << U)" begin
        t = 0.001
        U = 2.0 # U and t are close
        beta = 20.0 # Low temperature
        n_matsubara = 100

        # Flavors: 0 (site 1, up), 1 (site 1, down), 2 (site 2, up), 3 (site 2, down)
        # The analytical formula assumes mu=U/2, so we add the on-site energy term -U/2.
        hopping_str = """
        0 0 $(-U/2) 0.0
        1 1 $(-U/2) 0.0
        2 2 $(-U/2) 0.0
        3 3 $(-U/2) 0.0
        0 2 $(-t) 0.0
        2 0 $(-t) 0.0
        1 3 $(-t) 0.0
        3 1 $(-t) 0.0
        """
        coulomb_str = """
        4
        0 0 1 1 0 $(U) 0.0
        1 1 0 0 1 $(U) 0.0
        2 2 3 3 2 $(U) 0.0
        3 3 2 2 3 $(U) 0.0
        """

        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
        evals, evecs = solve_hamiltonian(H)

        # Calculate numerical Green's function
        iwn, g11_numeric = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)
        _, g12_numeric = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 3)

        # Analytical formula
        c = sqrt(16*t^2 + U^2)
        c_minus_U = c - U
        a_sq = (32*t^2 + 2*c_minus_U^2) / c_minus_U^2
        
        term1_factor = (1 + 4*t / c_minus_U)^2
        term2_factor = (1 - 4*t / c_minus_U)^2

        # G_11
        g11_analytical = (1 / (8 * a_sq)) .* (
            (term1_factor ./ (iwn .+ t .- c/2)) .+
            (term2_factor ./ (iwn .- t .- U/2)) .+
            (term1_factor ./ (iwn .- t .+ U/2)) .+
            (term2_factor ./ (iwn .+ t .+ U/2))
        ) .+ (3/16) .* (
            (1 ./ (iwn .+ t .- U/2)) .+
            (1 ./ (iwn .- t .- U/2)) .+
            (1 ./ (iwn .- t .+ U/2)) .+
            (1 ./ (iwn .+ t .+ U/2))
        )

        # G_12
        g12_analytical = (1 / (8 * a_sq)) .* (
            (-term1_factor ./ (iwn .+ t .- c/2)) .+
            (term2_factor ./ (iwn .- t .- U/2)) .+
            (term1_factor ./ (iwn .- t .+ U/2)) .+
            (-term2_factor ./ (iwn .+ t .+ U/2))
        ) .+ (3/16) .* (
            (1 ./ (iwn .+ t .- U/2)) .+
            (-1 ./ (iwn .- t .- U/2)) .+
            (-1 ./ (iwn .- t .+ U/2)) .+
            (1 ./ (iwn .+ t .+ U/2))
        )
        
        @test g11_numeric ≈ g11_analytical atol=1e-6
        @test g12_numeric ≈ g12_analytical atol=1e-6
    end
end