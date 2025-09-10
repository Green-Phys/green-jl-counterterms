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

@testset "Matsubara Green's Function (non-interacting)" begin
    # Non-interacting single-site model: H = ϵ c^† c
    ϵ = 0.5
    hopping_str = "0 0 $(ϵ) 0.0"
    coulomb_str = "0"

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    evals, evecs = solve_hamiltonian(H)

    beta = 10.0
    n_matsubara = 100
    iwn, giwn = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)

    # Analytical result: G(iω_n) = 1 / (iω_n - ϵ)
    giwn_analytical = 1 ./ (iwn .- ϵ)

    @test giwn ≈ giwn_analytical atol=1e-9
end

@testset "Imaginary-Time Green's Function (non-interacting)" begin
    # Non-interacting single-site model: H = ϵ c^† c
    ϵ = 0.5
    hopping_str = "0 0 $(ϵ) 0.0"
    coulomb_str = "0"

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    evals, evecs = solve_hamiltonian(H)

    beta = 10.0
    n_tau = 100
    tau_grid, gtau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1)

    # Analytical result: G(τ) = -exp(-ϵτ) / (1 + exp(-βϵ))
    Z_analytical = 1 + exp(-beta * ϵ)
    gtau_analytical = -exp.(-ϵ .* tau_grid) ./ Z_analytical

    @test gtau ≈ gtau_analytical atol=1e-9
end
