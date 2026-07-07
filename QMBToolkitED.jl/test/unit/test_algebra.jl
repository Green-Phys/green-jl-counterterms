"""
Unit Tests: Linear Algebra Operations

Tests for operator multiplication, addition, scalar multiplication, adjoint,
and conversion to/from matrices.
"""

using Test
using QMBToolkitED
using LinearAlgebra

@testset "Linear Algebra" begin

    @testset "Operator multiplication" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)

        c1_dag = c1'
        c2_dag = c2'

        n1 = c1_dag * c1
        n2 = c2_dag * c2

        @test n1.flux == QN_N(0)
        @test n2.flux == QN_N(0)
        @test !isempty(n1.blocks)
        @test !isempty(n2.blocks)
        @test n1.basis === basis
        @test n2.basis === basis
    end

    @testset "Operator addition" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)

        @test c1.flux == c2.flux

        c_sum = c1 + c2
        @test c_sum.flux == c1.flux
        @test c_sum.basis === basis
        @test !isempty(c_sum.blocks)
    end

    @testset "Operator subtraction" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)

        c_diff = c1 - c2
        @test c_diff.flux == c1.flux

        c_zero = c1 - c1
        @test isempty(c_zero.blocks) || all(iszero.(values(c_zero.blocks)))
    end

    @testset "Scalar multiplication" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)

        c1_scaled = 2.0 * c1
        @test c1_scaled.flux == c1.flux
        @test c1_scaled.basis === basis

        for (key, block) in c1.blocks
            @test c1_scaled.blocks[key] ≈ 2.0 * block
        end

        c1_complex = (1.0 + 2.0im) * c1
        @test c1_complex.flux == c1.flux

        c1_scaled2 = c1 * 2.0
        @test c1_scaled2.flux == c1.flux
        for (key, block) in c1_scaled.blocks
            @test c1_scaled2.blocks[key] ≈ block
        end

        c1_div = c1 / 2.0
        for (key, block) in c1.blocks
            @test c1_div.blocks[key] ≈ 0.5 * block
        end

        c1_zero = 0 * c1
        @test isempty(c1_zero.blocks)
    end

    @testset "Unary negation" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c1_neg = -c1
        @test c1_neg.flux == c1.flux

        for (key, block) in c1.blocks
            @test c1_neg.blocks[key] ≈ -block
        end
    end

    @testset "Adjoint operator" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c1_dag = c1'
        @test c1_dag.flux == QN_N(1)
        @test c1_dag.basis === basis

        c1_dagdag = c1_dag'
        @test c1_dagdag.flux == c1.flux

        for ((tgt, src), block) in c1.blocks
            @test haskey(c1_dag.blocks, (src, tgt))
            @test c1_dag.blocks[(src, tgt)] ≈ adjoint(block)
        end
    end

    @testset "Flux arithmetic" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        c = annihilators(basis)

        n1 = c[1]' * c[1]
        @test n1.flux == QN_N(0)

        n_total = c[1]' * c[1] + c[2]' * c[2]
        @test n_total.flux == QN_N(0)

        c1_scaled = 2.0 * c[1]
        @test c1_scaled.flux == c[1].flux

        c1_dag = c[1]'
        @test c1_dag.flux == QN_N(1)
    end

    @testset "Fermionic anticommutation relations" begin
        # 4-flavor system with QN_NSz (Hubbard dimer with spin)
        flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = Basis(flavor_qns)
        nflavors = 4
        c = annihilators(basis)
        c_dag = [op' for op in c]
        dim = basis.total_dim
        I_mat = Matrix{ComplexF64}(I, dim, dim)
        zero_mat = zeros(ComplexF64, dim, dim)

        # {c_i, c_j} = 0 for all i < j
        for i = 1:nflavors
            for j = (i+1):nflavors
                anticomm = to_matrix(c[i] * c[j] + c[j] * c[i])
                @test anticomm ≈ zero_mat atol=1e-14
            end
        end

        # {c_i, c_j†} = δ_ij I for all i, j
        for i = 1:nflavors
            for j = 1:nflavors
                anticomm = to_matrix(c[i] * c_dag[j] + c_dag[j] * c[i])
                if i == j
                    @test anticomm ≈ I_mat atol=1e-14
                else
                    @test anticomm ≈ zero_mat atol=1e-14
                end
            end
        end
    end

    @testset "Conversion to matrix" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c1_mat = to_matrix(c1)

        @test size(c1_mat) == (basis.total_dim, basis.total_dim)
        @test size(c1_mat) == (4, 4)
        # Annihilators now return Float64 since their values are ±1
        @test c1_mat isa Matrix{Float64}
    end

    @testset "Matrix conversion consistency" begin
        flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c1_dag = c1'
        n1 = c1_dag * c1
        n1_mat = to_matrix(n1)

        @test n1_mat ≈ adjoint(n1_mat) atol=1e-14

        eigs = eigvals(n1_mat)
        @test all(real(eigs) .>= -1e-14)
    end

    @testset "Operator algebra distributivity" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)
        c1_dag = c1'

        lhs = (c1 + c2) * c1_dag
        rhs = c1 * c1_dag + c2 * c1_dag

        @test to_matrix(lhs) ≈ to_matrix(rhs) atol=1e-14
    end

    @testset "Scalar multiplication distributivity" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)
        α = 2.0 + 3.0im

        lhs = α * (c1 + c2)
        rhs = α * c1 + α * c2

        @test to_matrix(lhs) ≈ to_matrix(rhs) atol=1e-14
    end

    @testset "Adjoint distributivity" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)

        # (c1 + c2)' = c1' + c2'
        @test to_matrix((c1 + c2)') ≈ to_matrix(c1' + c2') atol=1e-14

        # (c1 * c2)' = c2' * c1'
        @test to_matrix((c1 * c2)') ≈ to_matrix(c2' * c1') atol=1e-14
    end

    @testset "Total number operator" begin
        flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        N = sum([annihilator(basis, i)' * annihilator(basis, i) for i = 1:3])
        @test N.flux == QN_N(0)

        N_mat = to_matrix(N)
        @test N_mat ≈ adjoint(N_mat) atol=1e-14

        eigs = sort(real(eigvals(N_mat)))
        expected_eigs = [0, 1, 1, 1, 2, 2, 2, 3]
        @test eigs ≈ expected_eigs atol=1e-14
    end

    @testset "Operator on larger system" begin
        flavor_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        @test basis.total_dim == 16

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)

        hopping = c1' * c2 + c2' * c1
        @test hopping.flux == QN_N(0)

        hopping_mat = to_matrix(hopping)
        @test hopping_mat ≈ adjoint(hopping_mat) atol=1e-14
    end

end
