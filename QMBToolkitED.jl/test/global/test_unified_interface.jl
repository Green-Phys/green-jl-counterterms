"""
Unified Interface Tests: Verify method parameter functionality and backwards compatibility.

Tests that the unified interface:
1. Works with method=:dense (same results as original)
2. Works with method=:blockdense (same results as BlockDense module)
3. Maintains backwards compatibility (default method works)
4. Auto-detection works correctly
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra
using SparseArrays

@testset "Unified Interface Tests" begin

    @testset "get_annihilators - backwards compatibility" begin
        # Default behavior (no method specified) should work as before
        nflavors = 3
        c = get_annihilators(nflavors)

        @test length(c) == nflavors
        @test all(x -> isa(x, Matrix{ComplexF64}), c)

        # Check dimensions
        nfock = 1 << nflavors
        @test all(x -> size(x) == (nfock, nfock), c)
    end

    @testset "get_annihilators - explicit method=:dense" begin
        nflavors = 3
        c_default = get_annihilators(nflavors)
        c_dense = get_annihilators(nflavors; method=:dense)

        # Should be identical
        for i in 1:nflavors
            @test c_default[i] == c_dense[i]
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

        @test isa(H, Matrix)
        @test length(c) == 2
        @test all(x -> isa(x, Matrix{ComplexF64}), c)
    end

    @testset "build_hamiltonian - explicit method=:dense" begin
        hopping_input1 = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input1 = IOBuffer("2\n")
        H_default, c_default = build_hamiltonian(hopping_input1, coulomb_input1)

        hopping_input2 = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input2 = IOBuffer("2\n")
        H_dense, c_dense = build_hamiltonian(hopping_input2, coulomb_input2; method=:dense)

        # Should be identical
        @test H_default == H_dense
        for i in 1:length(c_default)
            @test c_default[i] == c_dense[i]
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

    @testset "solve_hamiltonian - auto-detection for SparseMatrixCSC" begin
        nflavors = 2
        c = get_annihilators(nflavors)
        H = sum(c[i]' * c[i] for i in 1:nflavors)

        evals_auto, evecs_auto = solve_hamiltonian(H)
        evals_dense, evecs_dense = solve_hamiltonian(H; method=:dense)

        @test evals_auto ≈ evals_dense atol=1e-14
        @test abs.(evecs_auto' * evecs_dense) ≈ Matrix{Float64}(I, size(evecs_auto, 2), size(evecs_dense, 2)) atol=1e-14
    end

    @testset "solve_hamiltonian - auto-detection for BlockDenseOperator" begin
        nflavors = 2
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c = get_annihilators(basis; method=:blockdense)
        H = sum(c[i]' * c[i] for i in 1:nflavors)

        evals_auto, evecs_auto = solve_hamiltonian(H)
        evals_block, evecs_block = solve_hamiltonian(H; method=:blockdense)

        @test evals_auto ≈ evals_block atol=1e-14
        @test abs.(evecs_auto' * evecs_block) ≈ Matrix{Float64}(I, size(evecs_auto, 2), size(evecs_block, 2)) atol=1e-14
    end

    @testset "solve_hamiltonian - error on wrong method" begin
        nflavors = 2

        # Sparse Hamiltonian
        c_dense = get_annihilators(nflavors)
        H_dense = sum(c_dense[i]' * c_dense[i] for i in 1:nflavors)
        @test_throws ErrorException solve_hamiltonian(H_dense; method=:blockdense)

        # Block-dense Hamiltonian
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators(basis; method=:blockdense)
        H_block = sum(c_block[i]' * c_block[i] for i in 1:nflavors)
        @test_throws ErrorException solve_hamiltonian(H_block; method=:dense)
    end

    @testset "Consistency: dense vs blockdense eigenvalues" begin
        # Compare dense and blockdense methods on same Hamiltonian
        nflavors = 3

        # Sparse version
        hopping_input1 = IOBuffer("0 1 1.0 0.0\n1 2 1.0 0.0\n")
        coulomb_input1 = IOBuffer("3\n")
        H_dense, _ = build_hamiltonian(hopping_input1, coulomb_input1; method=:dense)
        evals_dense, _ = solve_hamiltonian(H_dense)

        # Block-dense version
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        hopping_input2 = IOBuffer("0 1 1.0 0.0\n1 2 1.0 0.0\n")
        coulomb_input2 = IOBuffer("3\n")
        H_block, _ = build_hamiltonian(hopping_input2, coulomb_input2; method=:blockdense, basis=basis)
        evals_block, _ = solve_hamiltonian(H_block)

        # Eigenvalues should match
        @test sort(evals_dense) ≈ sort(evals_block) atol=1e-12
    end

    @testset "Consistency: anticommutation relations" begin
        nflavors = 3

        # Sparse version
        c_dense = get_annihilators(nflavors; method=:dense)

        # Block-dense version
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators(basis; method=:blockdense)

        # Check {c_i, c_j'} = δ_ij I for both methods
        for i in 1:nflavors, j in 1:nflavors
            # Dense method
            anticomm_dense = c_dense[i] * c_dense[j]' + c_dense[j]' * c_dense[i]

            # BlockDense method
            anticomm_block = c_block[i] * c_block[j]' + c_block[j]' * c_block[i]
            anticomm_block_dense = to_dense(anticomm_block)

            if i == j
                I_mat = Matrix{ComplexF64}(I, basis.total_dim, basis.total_dim)
                @test anticomm_dense ≈ I_mat atol=1e-14
                @test anticomm_block_dense ≈ I_mat atol=1e-14
            else
                @test iszero(anticomm_dense)
                @test iszero(anticomm_block_dense)
            end
        end
    end

    @testset "Output type consistency" begin
        # Verify that method=:dense returns dense types
        nflavors = 2
        c_dense = get_annihilators(nflavors; method=:dense)
        @test all(x -> isa(x, Matrix), c_dense)

        hopping_input = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input = IOBuffer("2\n")
        H_dense, _ = build_hamiltonian(hopping_input, coulomb_input; method=:dense)
        @test isa(H_dense, Matrix)

        evals_dense, evecs_dense = solve_hamiltonian(H_dense)
        @test isa(evals_dense, Vector{Float64})
        @test isa(evecs_dense, Matrix{ComplexF64})

        # Verify that method=:blockdense returns block-dense types
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators(basis; method=:blockdense)
        @test all(x -> isa(x, BlockDenseOperator), c_block)

        hopping_input2 = IOBuffer("0 1 1.0 0.0\n")
        coulomb_input2 = IOBuffer("2\n")
        H_block, _ = build_hamiltonian(hopping_input2, coulomb_input2; method=:blockdense, basis=basis)
        @test isa(H_block, BlockDenseOperator)

        evals_block, evecs_block = solve_hamiltonian(H_block)
        @test isa(evals_block, Vector{Float64})
        @test isa(evecs_block, Matrix{ComplexF64})
    end

end

println("\nUnified Interface Tests Complete ✓")
