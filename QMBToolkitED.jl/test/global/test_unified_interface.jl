"""
Unified Interface Tests: Verify method parameter functionality.

Tests that the unified interface:
1. Works with method=:dense (same results as original)
2. Works with method=:blockdense (same results as BlockDense module)
3. Auto-detection works correctly
"""

using Test
using QMBToolkitED
using QMBToolkitED.Dense
using QMBToolkitED.BlockDense
using LinearAlgebra
using .GlobalTestUtils

@testset "Unified Interface Tests" begin

    @testset "get_annihilators - backwards compatibility" begin
        # Default behavior (no method specified) should work as before
        nflavors = 3
        c = get_annihilators(nflavors)

        @test length(c) == nflavors
        @test all(x -> isa(x, DenseOperator), c)

        # Check dimensions
        nfock = 1 << nflavors
        @test all(x -> size(x) == (nfock, nfock), c)
    end

    @testset "get_annihilators - explicit method=:dense" begin
        nflavors = 3
        c_default = get_annihilators(nflavors)
        c_dense = get_annihilators(nflavors; method=:dense)

        # Should be identical (DenseOperators)
        for i in 1:nflavors
            @test to_matrix(c_default[i]) == to_matrix(c_dense[i])
        end
    end

    @testset "get_annihilators - method=:blockdense" begin
        nflavors = 3
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators(basis; method=:blockdense)

        @test length(c_block) == nflavors
        @test all(x -> isa(x, BlockDenseOperator), c_block)

        # Check flux
        for i in 1:nflavors
            @test c_block[i].flux.delta_qn == QN_N(-1)
        end
    end

    @testset "get_annihilators - error on wrong method" begin
        nflavors = 3
        basis = make_basis([QN_N(1) for _ in 1:nflavors])

        # Can't use :blockdense method with Int
        @test_throws ErrorException get_annihilators(nflavors; method=:blockdense)

        # Can't use :dense method with BlockDenseBasis
        @test_throws ErrorException get_annihilators(basis; method=:dense)
    end

    @testset "build_hamiltonian - backwards compatibility" begin
        # Create simple test inputs
        hopping_input = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input = IOBuffer("2\n")

        H, c = build_hamiltonian(hopping_input, coulomb_input)

        @test isa(H, DenseOperator)
        @test length(c) == 2
        @test all(x -> isa(x, DenseOperator), c)
    end

    @testset "build_hamiltonian - explicit method=:dense" begin
        hopping_input1 = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input1 = IOBuffer("2\n")
        H_default, c_default = build_hamiltonian(hopping_input1, coulomb_input1)

        hopping_input2 = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input2 = IOBuffer("2\n")
        H_dense, c_dense = build_hamiltonian(hopping_input2, coulomb_input2; method=:dense)

        # Should be identical (use to_matrix to compare underlying matrices)
        @test to_matrix(H_default) == to_matrix(H_dense)
        for i in 1:length(c_default)
            @test to_matrix(c_default[i]) == to_matrix(c_dense[i])
        end
    end

    @testset "build_hamiltonian - method=:blockdense" begin
        nflavors = 2
        basis = make_basis([QN_N(1) for _ in 1:nflavors])

        hopping_input = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input = IOBuffer("2\n")
        H_block, c_block = build_hamiltonian(hopping_input, coulomb_input; method=:blockdense, basis=basis)

        @test isa(H_block, BlockDenseOperator)
        @test length(c_block) == nflavors
        @test all(x -> isa(x, BlockDenseOperator), c_block)

        # Check Hamiltonian is number-conserving
        @test H_block.flux.delta_qn == QN_N(0)
    end

    @testset "build_hamiltonian - error without basis for blockdense" begin
        hopping_input = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input = IOBuffer("2\n")

        @test_throws ErrorException build_hamiltonian(hopping_input, coulomb_input; method=:blockdense)
    end

    @testset "solve_hamiltonian - auto-detection for DenseOperator" begin
        nflavors = 2
        c = get_annihilators(nflavors)
        H = sum(c[i]' * c[i] for i in 1:nflavors)

        result_auto = solve_hamiltonian(H)
        result_dense = solve_hamiltonian(H; method=:dense)

        evals_auto = get_evals(result_auto)
        evals_dense = get_evals(result_dense)
        evecs_auto = to_matrix(result_auto.eigenvectors)
        evecs_dense = to_matrix(result_dense.eigenvectors)

        @test evals_auto ≈ evals_dense atol=1e-14
        @test abs.(evecs_auto' * evecs_dense) ≈ Matrix{Float64}(I, size(evecs_auto, 2), size(evecs_dense, 2)) atol=1e-14
    end

    @testset "solve_hamiltonian - auto-detection for BlockDenseOperator" begin
        nflavors = 2
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c = get_annihilators(basis; method=:blockdense)
        H = sum(c[i]' * c[i] for i in 1:nflavors)

        result_auto = solve_hamiltonian(H)
        result_block = solve_hamiltonian(H; method=:blockdense)

        evals_auto = get_evals(result_auto)
        evals_block = get_evals(result_block)
        evecs_auto = to_matrix(result_auto.eigenvectors)
        evecs_block = to_matrix(result_block.eigenvectors)

        @test evals_auto ≈ evals_block atol=1e-14
        @test abs.(evecs_auto' * evecs_block) ≈ Matrix{Float64}(I, size(evecs_auto, 2), size(evecs_block, 2)) atol=1e-14
    end

    @testset "solve_hamiltonian - error on wrong method" begin
        nflavors = 2

        # DenseOperator Hamiltonian
        c_dense = get_annihilators(nflavors)
        H_dense = sum(c_dense[i]' * c_dense[i] for i in 1:nflavors)
        @test_throws ErrorException solve_hamiltonian(H_dense; method=:blockdense)

        # Block-dense Hamiltonian
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators(basis; method=:blockdense)
        H_block = sum(c_block[i]' * c_block[i] for i in 1:nflavors)
        @test_throws ErrorException solve_hamiltonian(H_block; method=:dense)
    end

    @testset "Result Types and Field Access" begin
        # Test that result types have correct fields and eigenvalue/eigenvector access works
        U = 2.0
        hopping_str = """
        0 0 $(-U/2) 0.0
        1 1 $(-U/2) 0.0
        """
        coulomb_str = """
        2
        0 0 1 1 0 $(U) 0.0
        1 1 0 0 1 $(U) 0.0
        """

        # Dense method
        H_dense, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
        result_dense = solve_hamiltonian(H_dense)

        # Test result types
        @test isa(result_dense, DenseEigenResult)
        @test isa(result_dense.eigenvalues, DenseDiagonalOperator)
        @test isa(result_dense.eigenvectors, DenseOperator)

        # Test eigenvalue access
        evals_dense = get_evals(result_dense)
        @test isa(evals_dense, Vector{Float64})
        @test length(evals_dense) == 4  # 2^2 = 4 states

        # Test eigenvector access
        evecs_dense = to_matrix(result_dense.eigenvectors)
        @test size(evecs_dense) == (4, 4)

        # BlockDense method
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        result_block = solve_hamiltonian(H_block)

        # Test result types
        @test isa(result_block, BlockEigenResult)
        @test isa(result_block.eigenvalues, BlockDenseDiagonalOperator)
        @test isa(result_block.eigenvectors, BlockDenseOperator)

        # Test eigenvalue access
        evals_block = get_evals(result_block)
        @test isa(evals_block, Vector{Float64})
        @test length(evals_block) == 4  # Same total Hilbert space dim

        # Verify eigenvectors are correctly reconstructed from sectors
        total_dim = basis.total_dim
        evecs_block = to_matrix(result_block.eigenvectors)
        @test size(evecs_block) == (total_dim, total_dim)

        # Verify eigenvectors are orthonormal
        @test evecs_block' * evecs_block ≈ Matrix{ComplexF64}(I, total_dim, total_dim) atol=1e-12

        # Verify eigenvalues and eigenvectors satisfy H * v = λ * v
        for (sector_idx, diag) in result_block.eigenvalues.sector_diags
            U_sector = result_block.eigenvectors.blocks[(sector_idx, sector_idx)]
            # H_sector may be missing if it's zero (e.g., vacuum sector)
            H_sector = get(H_block.blocks, (sector_idx, sector_idx),
                           zeros(ComplexF64, size(U_sector, 1), size(U_sector, 1)))
            for i in 1:length(diag)
                Hv = H_sector * U_sector[:, i]
                λv = diag[i] * U_sector[:, i]
                @test Hv ≈ λv atol=1e-10
            end
        end

        # Cross-validate: Dense and BlockDense eigenvalues should match
        @test sort(evals_dense) ≈ sort(evals_block) atol=1e-12

        # Eigenvectors should also match (up to phase/ordering)
        overlap = abs.(evecs_dense' * evecs_block)
        # Each row/column should have exactly one entry ≈ 1
        @test all(maximum(overlap, dims=2) .≈ 1.0)
        @test all(maximum(overlap, dims=1) .≈ 1.0)
    end

    @testset "Output type consistency" begin
        # Verify that method=:dense returns DenseOperator types
        nflavors = 2
        c_dense = get_annihilators(nflavors; method=:dense)
        @test all(x -> isa(x, DenseOperator), c_dense)

        hopping_input = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input = IOBuffer("2\n")
        H_dense, _ = build_hamiltonian(hopping_input, coulomb_input; method=:dense)
        @test isa(H_dense, DenseOperator)

        result_dense = solve_hamiltonian(H_dense)
        @test isa(result_dense, DenseEigenResult)
        @test isa(result_dense.eigenvalues, DenseDiagonalOperator)
        @test isa(result_dense.eigenvectors, DenseOperator)

        # Verify that method=:blockdense returns block-dense types
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators(basis; method=:blockdense)
        @test all(x -> isa(x, BlockDenseOperator), c_block)

        hopping_input2 = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input2 = IOBuffer("2\n")
        H_block, _ = build_hamiltonian(hopping_input2, coulomb_input2; method=:blockdense, basis=basis)
        @test isa(H_block, BlockDenseOperator)

        result_block = solve_hamiltonian(H_block)
        @test isa(result_block, BlockEigenResult)
        @test isa(result_block.eigenvalues, BlockDenseDiagonalOperator)
        @test isa(result_block.eigenvectors, BlockDenseOperator)
    end

end

println("\nUnified Interface Tests Complete ✓")
