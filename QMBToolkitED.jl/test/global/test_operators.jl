"""
Global Tests: Fermionic Operators

Tests fermionic operator construction and anticommutation relations.
These tests verify the user-facing API works correctly for both Dense and BlockDense methods.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra
using .GlobalTestUtils

@testset "Operators: Dense Method" begin
    nflavors = 6
    c = get_annihilators(nflavors)
    c_dag = [op' for op in c]
    nfock = 1 << nflavors

    verify_anticommutation(c, c_dag, nfock)
end

@testset "Operators: BlockDense Method with QN_Trivial" begin
    nflavors = 6
    site_qns = [QN_Trivial() for _ in 1:nflavors]
    basis = make_basis(site_qns)
    c = get_annihilators_blockdense(basis)
    c_dag = [op' for op in c]

    # Should have exactly 1 sector with full Hilbert space dimension
    @test length(basis.sectors) == 1
    @test basis.sectors[1].dim == basis.total_dim

    verify_anticommutation(c, c_dag, basis.total_dim)
end

@testset "Operators: BlockDense Method with QN_N" begin
    nflavors = 6
    site_qns = [QN_N(1) for _ in 1:nflavors]
    basis = make_basis(site_qns)
    c = get_annihilators_blockdense(basis)
    c_dag = [op' for op in c]

    verify_anticommutation(c, c_dag, basis.total_dim)
end

@testset "Operators: BlockDense Method with QN_NSz" begin
    # 3 sites with spin: [↑, ↓, ↑, ↓, ↑, ↓]
    nflavors = 6
    site_qns = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:nflavors]
    basis = make_basis(site_qns)
    c = get_annihilators_blockdense(basis)
    c_dag = [op' for op in c]

    verify_anticommutation(c, c_dag, basis.total_dim)
end