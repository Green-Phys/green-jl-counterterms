
"""
BlockDense Algebra Tests

Tests block-dense operator algebra and conversions.
"""

# This file is included by runtests.jl, so it uses the environment setup there.
# If running standalone, uncomment the following:
# using Test
# using QMBToolkitED.BlockDense
# using LinearAlgebra

# We'll use the detailed tests in the 'detailed' directory
include("detailed/test_linalg.jl")


@testset "BlockDense: Block Algebra" begin

    @testset "Flux Arithmetic" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        c = get_annihilators_blockdense(basis)

        # Multiplication composes flux
        n1 = c[1]' * c[1]
        @test n1.flux.delta_qn == QN_N(0)  # QN(1) + QN(-1) = QN(0)

        # Addition requires matching flux
        n_total = c[1]' * c[1] + c[2]' * c[2]
        @test n_total.flux.delta_qn == QN_N(0)

        # Scalar multiplication preserves flux
        c1_scaled = 2.0 * c[1]
        @test c1_scaled.flux == c[1].flux

        # Adjoint reverses flux
        c1_dag = c[1]'
        @test c1_dag.flux.delta_qn == QN_N(1)
    end

    @testset "Block-to-Dense Conversion" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        c = get_annihilators_blockdense(basis)

        # Conversion produces correct type and size
        c1_dense = to_dense(c[1])
        @test c1_dense isa Matrix{ComplexF64}
        @test size(c1_dense) == (basis.total_dim, basis.total_dim)
    end

end
