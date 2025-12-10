"""
Global Tests: Fermionic Operators

Tests fermionic operator construction and anticommutation relations.
These tests verify the user-facing API works correctly for both Dense and BlockDense methods.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra

@testset "Operators: Dense Method" begin
    nflavors = 6
    c = get_annihilators(nflavors)
    c_dag = [op' for op in c]
    nfock = 1 << nflavors

    # Test {c_i, c_j} = 0 for i != j
    @testset "Anticommutator {c_i, c_j}" begin
        for i in 1:nflavors
            for j in (i+1):nflavors
                @test c[i] * c[j] + c[j] * c[i] ≈ zeros(ComplexF64, nfock, nfock) atol=1e-9
            end
        end
    end

    # Test {c_i, c_j†} = δ_ij * I
    @testset "Anticommutator {c_i, c_j†}" begin
        for i in 1:nflavors
            for j in 1:nflavors
                result = c[i] * c_dag[j] + c_dag[j] * c[i]
                if i == j
                    @test result ≈ Matrix{ComplexF64}(I, nfock, nfock) atol=1e-9
                else
                    @test result ≈ zeros(ComplexF64, nfock, nfock) atol=1e-9
                end
            end
        end
    end
end

@testset "Operators: BlockDense Method" begin
    nflavors = 6
    site_qns = [QN_N(1) for _ in 1:nflavors]
    basis = make_basis(site_qns)
    c = get_annihilators_blockdense(basis)
    c_dag = [op' for op in c]

    # Test {c_i, c_j} = 0 for i != j
    @testset "Anticommutator {c_i, c_j}" begin
        for i in 1:nflavors
            for j in (i+1):nflavors
                result = to_dense(c[i] * c[j] + c[j] * c[i])
                @test result ≈ zeros(ComplexF64, basis.total_dim, basis.total_dim) atol=1e-9
            end
        end
    end

    # Test {c_i, c_j†} = δ_ij * I
    @testset "Anticommutator {c_i, c_j†}" begin
        for i in 1:nflavors
            for j in 1:nflavors
                result = to_dense(c[i] * c_dag[j] + c_dag[j] * c[i])
                if i == j
                    @test result ≈ Matrix{ComplexF64}(I, basis.total_dim, basis.total_dim) atol=1e-9
                else
                    @test result ≈ zeros(ComplexF64, basis.total_dim, basis.total_dim) atol=1e-9
                end
            end
        end
    end
end
