using QMBToolkitED
using Test
using LinearAlgebra
using SparseArrays

@testset "Fermionic Operators" begin
    nflavours = 4 # A slightly larger test case
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


@testset "Hamiltonian from files" begin
    hopping_file = "hopping.txt"
    coulomb_file = "coulomb.txt"
    
    # 1. Build Hamiltonian
    H, annh = build_hamiltonian(hopping_file, coulomb_file)
    nflavours = 2
    nfock = 1 << nflavours
    @test size(H) == (nfock, nfock)
    @test ishermitian(H)

    # Check some matrix elements
    H_dense = Matrix(H)
    # H = 1.0 * (c_0^† c_1 + c_1^† c_0) + 0.5 * c_0^† c_1^† c_1 c_0
    # States: |00>, |01>, |10>, |11> (vacuum, c1 occ, c0 occ, c0c1 occ)
    # Indices: 1,    2,    3,    4
    # Expected: H[2,3] = 1.0, H[3,2] = 1.0
    #           H[4,4] = 0.5
    @test H_dense[2,3] ≈ 1.0
    @test H_dense[3,2] ≈ 1.0
    @test H_dense[4,4] ≈ 0.5

    # 2. Solve Hamiltonian
    evals, evecs = solve_hamiltonian(H)
    @test length(evals) == nfock
    # Eigenvalues of [[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0.5]] are -1, 0, 0.5, 1
    # My matrix is [[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0.5]]
    # The eigenvalues of [[0,1],[1,0]] are -1, 1.
    # So evals should be -1, 1, 0, 0.5.
    # The matrix is for block with N=1 particle.
    # The full matrix is 4x4.
    # N=0 block: H=0. Eval=0.
    # N=2 block: H=0.5. Eval=0.5.
    # N=1 block: [[0, 1], [1, 0]]. Evals = -1, 1.
    @test sort(real(evals)) ≈ [-1.0, 0.0, 0.5, 1.0] atol=1e-9

    # 3. Calculate Green's function
    beta = 10.0
    tau_grid, gtau = calculate_greens_function(evals, evecs, annh, beta, 100, 1, 1)
    @test length(gtau) == 100
end
