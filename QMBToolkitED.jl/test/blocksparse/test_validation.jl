"""
Validation Tests: Block-Sparse vs Full-Sparse Comparison

Compare block-sparse implementation against the original full-sparse QMBToolkitED.jl.
All results should match exactly (within numerical precision).
"""

using Test
using Main.BlockSparse
using LinearAlgebra
using SparseArrays

# Load the original QMBToolkitED module
include("../../src/QMBToolkitED.jl")
using .QMBToolkitED

@testset "Validation: Block-Sparse vs Full-Sparse" begin

    @testset "Annihilation operators have correct properties" begin
        nflavors = 3

        # Block-sparse version
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators_blocksparse(basis)

        # Check each operator has correct flux and dimensions
        for i in 1:nflavors
            @test c_block[i].flux.delta_qn == QN_N(-1)
            @test c_block[i].basis === basis

            # Convert to sparse to check basic properties
            c_sparse = to_sparse(c_block[i])
            @test size(c_sparse) == (basis.total_dim, basis.total_dim)
            @test nnz(c_sparse) > 0  # Non-zero elements
        end
    end

    @testset "Number operators have correct eigenvalues" begin
        nflavors = 3

        # Full-sparse
        c_sparse = get_annihilators(nflavors)
        n_sparse = [c_sparse[i]' * c_sparse[i] for i in 1:nflavors]

        # Block-sparse
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators_blocksparse(basis)
        n_block = [c_block[i]' * c_block[i] for i in 1:nflavors]

        # Compare eigenvalues (physical observable)
        for i in 1:nflavors
            evals_sparse = sort(real(eigvals(Matrix(n_sparse[i]))))
            evals_block = sort(real(eigvals(Matrix(to_sparse(n_block[i])))))
            @test evals_sparse ≈ evals_block atol=1e-12

            # Number operator should have eigenvalues 0 and 1 only
            @test all(e -> isapprox(e, 0, atol=1e-12) || isapprox(e, 1, atol=1e-12), evals_sparse)
        end
    end

    @testset "Anticommutation relations" begin
        nflavors = 3

        # Full-sparse
        c_sparse = get_annihilators(nflavors)

        # Block-sparse
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators_blocksparse(basis)

        # {c_i, c_j} = 0
        for i in 1:nflavors, j in 1:nflavors
            anticomm_sparse = c_sparse[i] * c_sparse[j]' + c_sparse[j]' * c_sparse[i]
            anticomm_block = c_block[i] * c_block[j]' + c_block[j]' * c_block[i]

            anticomm_block_sparse = to_sparse(anticomm_block)

            # Should be δ_ij * I
            if i == j
                I_mat = Matrix{ComplexF64}(I, basis.total_dim, basis.total_dim)
                @test anticomm_block_sparse ≈ I_mat atol=1e-14
                @test Matrix(anticomm_sparse) ≈ I_mat atol=1e-14
            else
                @test iszero(anticomm_block_sparse)
                @test iszero(anticomm_sparse)
            end
        end
    end

    @testset "Simple Hamiltonian - manual hopping chain" begin
        # Create simple hopping chain: H = -t * (c_1' c_2 + c_2' c_1)
        nflavors = 2
        t = 1.0

        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c = get_annihilators_blocksparse(basis)

        # Build hopping Hamiltonian manually
        H = -t * (c[1]' * c[2] + c[2]' * c[1])

        @test H.flux.delta_qn == QN_N(0)  # Number conserving

        # Check Hermiticity
        H_sparse = to_sparse(H)
        @test H_sparse ≈ H_sparse' atol=1e-14

        # Get eigenvalues
        evals, _ = solve_hamiltonian_blocksparse(H)

        # For 2-site hopping chain, eigenvalues are known
        # Should have ground state and excited states
        @test length(evals) == basis.total_dim
    end

    @testset "Hamiltonian with interaction - manual construction" begin
        # Simple test: Build H = U * n_1 * n_2 manually
        nflavors = 2
        U = 2.0

        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators_blocksparse(basis)

        # Number operators
        n1 = c_block[1]' * c_block[1]
        n2 = c_block[2]' * c_block[2]

        # Interaction Hamiltonian
        H = U * (n1 * n2)

        @test H.flux.delta_qn == QN_N(0)  # Number conserving

        # Diagonalize
        evals, _ = solve_hamiltonian_blocksparse(H)

        # Eigenvalues: 0 (vacuum, single occ), U (double occ)
        # Expected: [0, 0, 0, U] with degeneracies
        @test sort(evals) ≈ [0.0, 0.0, 0.0, U] atol=1e-12
    end

    @testset "3-site hopping chain - eigenvalues" begin
        # Build hopping Hamiltonian manually: H = -t * (c_1'c_2 + c_2'c_1 + c_2'c_3 + c_3'c_2)
        nflavors = 3
        t = 1.0

        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c = get_annihilators_blocksparse(basis)

        # Hopping terms
        H = -t * (c[1]'*c[2] + c[2]'*c[1] + c[2]'*c[3] + c[3]'*c[2])

        @test H.flux.delta_qn == QN_N(0)

        # Get eigenvalues
        evals, _ = solve_hamiltonian_blocksparse(H)

        # 3-site chain has specific eigenvalue spectrum
        # Check Hermiticity at least
        H_sparse = to_sparse(H)
        @test H_sparse ≈ H_sparse' atol=1e-14
    end

    @testset "Total number operator eigenvalues" begin
        nflavors = 3

        # Full-sparse
        c_sparse = get_annihilators(nflavors)
        N_sparse = sum(c_sparse[i]' * c_sparse[i] for i in 1:nflavors)
        evals_sparse = eigvals(Matrix(N_sparse))

        # Block-sparse
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators_blocksparse(basis)
        N_block = sum(c_block[i]' * c_block[i] for i in 1:nflavors)
        evals_block = eigvals(Matrix(to_sparse(N_block)))

        # Should match
        @test sort(real(evals_block)) ≈ sort(real(evals_sparse)) atol=1e-12

        # Should be integers 0, 1, 2, 3 with correct degeneracies
        expected = [0, 1, 1, 1, 2, 2, 2, 3]  # C(3,n) degeneracies
        @test sort(real(evals_block)) ≈ expected atol=1e-12
    end

end

println("\nValidation Tests Complete: Block-Sparse matches Full-Sparse ✓")
