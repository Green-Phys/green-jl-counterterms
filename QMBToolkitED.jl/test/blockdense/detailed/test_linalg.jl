"""
BlockDense Tests: Linear Algebra Operations

Tests for operator multiplication, addition, scalar multiplication, adjoint,
and conversion to/from matrices.
"""

using Test
using Main.BlockDense
using LinearAlgebra

@testset "BlockDense: Linear Algebra" begin

    @testset "Operator multiplication" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        # Creation operators
        c1_dag = c1'
        c2_dag = c2'

        # Number operators c†c should have flux = 0
        n1 = c1_dag * c1
        n2 = c2_dag * c2

        @test n1.flux.delta_qn == QN_N(0)
        @test n2.flux.delta_qn == QN_N(0)

        # Check blocks exist
        @test !isempty(n1.blocks)
        @test !isempty(n2.blocks)

        # Both operators act on same basis
        @test n1.basis === basis
        @test n2.basis === basis
    end

    @testset "Operator addition" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        # Both have same flux, can be added
        @test c1.flux.delta_qn == c2.flux.delta_qn

        c_sum = c1 + c2
        @test c_sum.flux.delta_qn == c1.flux.delta_qn
        @test c_sum.basis === basis

        # Check that blocks combine correctly
        @test !isempty(c_sum.blocks)
    end

    @testset "Operator subtraction" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        c_diff = c1 - c2
        @test c_diff.flux.delta_qn == c1.flux.delta_qn

        # Check c1 - c1 = 0
        c_zero = c1 - c1
        @test isempty(c_zero.blocks) || all(iszero.(values(c_zero.blocks)))
    end

    @testset "Scalar multiplication" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)

        # Scale by real number
        c1_scaled = 2.0 * c1
        @test c1_scaled.flux.delta_qn == c1.flux.delta_qn
        @test c1_scaled.basis === basis

        # Check block values scaled
        for (key, block) in c1.blocks
            @test c1_scaled.blocks[key] ≈ 2.0 * block
        end

        # Scale by complex number
        c1_complex = (1.0 + 2.0im) * c1
        @test c1_complex.flux.delta_qn == c1.flux.delta_qn

        # Commutative
        c1_scaled2 = c1 * 2.0
        @test c1_scaled2.flux.delta_qn == c1.flux.delta_qn
        for (key, block) in c1_scaled.blocks
            @test c1_scaled2.blocks[key] ≈ block
        end

        # Division
        c1_div = c1 / 2.0
        for (key, block) in c1.blocks
            @test c1_div.blocks[key] ≈ 0.5 * block
        end

        # Zero scaling
        c1_zero = 0 * c1
        @test isempty(c1_zero.blocks)
    end

    @testset "Unary negation" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)

        c1_neg = -c1
        @test c1_neg.flux.delta_qn == c1.flux.delta_qn

        for (key, block) in c1.blocks
            @test c1_neg.blocks[key] ≈ -block
        end
    end

    @testset "Adjoint operator" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)

        # c1 has flux QN_N(-1), c1' should have flux QN_N(1)
        c1_dag = c1'
        @test c1_dag.flux.delta_qn == QN_N(1)
        @test c1_dag.basis === basis

        # Double adjoint should return original
        c1_dagdag = c1_dag'
        @test c1_dagdag.flux.delta_qn == c1.flux.delta_qn

        # Check blocks are conjugate transposes with swapped indices
        for ((tgt, src), block) in c1.blocks
            @test haskey(c1_dag.blocks, (src, tgt))
            @test c1_dag.blocks[(src, tgt)] ≈ adjoint(block)
        end
    end

    @testset "Fermionic anticommutation {c_i, c_j} = 0" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)
        c3 = get_annihilator_blockdense(basis, 3)

        # {c_i, c_j} = c_i * c_j + c_j * c_i = 0
        anticomm_12 = c1 * c2 + c2 * c1
        anticomm_13 = c1 * c3 + c3 * c1
        anticomm_23 = c2 * c3 + c3 * c2

        # Should be zero operators (empty blocks or all zeros)
        @test isempty(anticomm_12.blocks) || all(iszero, values(anticomm_12.blocks))
        @test isempty(anticomm_13.blocks) || all(iszero, values(anticomm_13.blocks))
        @test isempty(anticomm_23.blocks) || all(iszero, values(anticomm_23.blocks))
    end

    @testset "Fermionic anticommutation {c_i, c_j'} = δ_ij" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        c1_dag = c1'
        c2_dag = c2'

        # {c_1, c_1'} = c_1 * c_1' + c_1' * c_1 = I (identity on full space)
        anticomm_11 = c1 * c1_dag + c1_dag * c1
        @test anticomm_11.flux.delta_qn == QN_N(0)

        # Convert to matrix to check it's identity
        anticomm_11_mat = to_matrix(anticomm_11)
        I_matrix = Matrix{ComplexF64}(I, basis.total_dim, basis.total_dim)
        @test anticomm_11_mat ≈ I_matrix atol=1e-14

        # {c_1, c_2'} = 0 (different orbitals)
        anticomm_12 = c1 * c2_dag + c2_dag * c1
        @test anticomm_12.flux.delta_qn == QN_N(0)

        anticomm_12_mat = to_matrix(anticomm_12)
        @test iszero(anticomm_12_mat)
    end

    @testset "Conversion to matrix" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)

        # Convert to matrix
        c1_mat = to_matrix(c1)

        # Check dimensions
        @test size(c1_mat) == (basis.total_dim, basis.total_dim)
        @test size(c1_mat) == (4, 4)  # 2^2 = 4

        # Should be Matrix type
        @test c1_mat isa Matrix{ComplexF64}
    end

    @testset "Conversion from matrix" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)

        # Round trip: block -> matrix -> block
        c1_mat = to_matrix(c1)
        c1_reconstructed = from_matrix(c1_mat, basis, c1.flux)

        # Should have same flux
        @test c1_reconstructed.flux.delta_qn == c1.flux.delta_qn

        # Should have same blocks (up to numerical precision)
        @test Set(keys(c1.blocks)) == Set(keys(c1_reconstructed.blocks))

        for key in keys(c1.blocks)
            @test c1.blocks[key] ≈ c1_reconstructed.blocks[key] atol=1e-14
        end
    end

    @testset "Matrix conversion consistency" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        c1_dag = c1'
        c2_dag = c2'

        # Number operator
        n1 = c1_dag * c1

        # Convert to matrix
        n1_mat = to_matrix(n1)

        # Should be Hermitian (up to numerical precision)
        @test n1_mat ≈ adjoint(n1_mat) atol=1e-14

        # Should have non-negative eigenvalues (number operator)
        eigs = eigvals(n1_mat)
        @test all(real(eigs) .>= -1e-14)  # Allow small numerical errors
    end

    @testset "Operator algebra distributivity" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        # (c1 + c2) * c1' = c1 * c1' + c2 * c1'
        c1_dag = c1'

        lhs = (c1 + c2) * c1_dag
        rhs = c1 * c1_dag + c2 * c1_dag

        lhs_mat = to_matrix(lhs)
        rhs_mat = to_matrix(rhs)

        @test lhs_mat ≈ rhs_mat atol=1e-14
    end

    @testset "Scalar multiplication distributivity" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        α = 2.0 + 3.0im

        # α * (c1 + c2) = α * c1 + α * c2
        lhs = α * (c1 + c2)
        rhs = α * c1 + α * c2

        lhs_mat = to_matrix(lhs)
        rhs_mat = to_matrix(rhs)

        @test lhs_mat ≈ rhs_mat atol=1e-14
    end

    @testset "Adjoint distributivity" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        # (c1 + c2)' = c1' + c2'
        lhs = (c1 + c2)'
        rhs = c1' + c2'

        lhs_mat = to_matrix(lhs)
        rhs_mat = to_matrix(rhs)

        @test lhs_mat ≈ rhs_mat atol=1e-14

        # (c1 * c2)' = c2' * c1' (note reversed order)
        c12 = c1 * c2
        c12_dag = c12'

        c2_dag = c2'
        c1_dag = c1'
        c2dag_c1dag = c2_dag * c1_dag

        c12_dag_mat = to_matrix(c12_dag)
        c2dag_c1dag_mat = to_matrix(c2dag_c1dag)

        @test c12_dag_mat ≈ c2dag_c1dag_mat atol=1e-14
    end

    @testset "Total number operator" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        # Total number operator: N = Σ_i c_i' * c_i
        N = sum([get_annihilator_blockdense(basis, i)' * get_annihilator_blockdense(basis, i)
                 for i in 1:3])

        @test N.flux.delta_qn == QN_N(0)

        # Convert to matrix and check eigenvalues
        N_mat = to_matrix(N)

        # Should be Hermitian
        @test N_mat ≈ adjoint(N_mat) atol=1e-14

        # Eigenvalues should be 0, 1, 2, 3 (particle numbers)
        eigs = sort(real(eigvals(N_mat)))
        expected_eigs = [0, 1, 1, 1, 2, 2, 2, 3]  # Degeneracies for 3 orbitals

        @test eigs ≈ expected_eigs atol=1e-14
    end

    @testset "Operator on larger system" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        @test basis.total_dim == 16  # 2^4

        # Create hopping term: c_1' * c_2 + c_2' * c_1
        c1 = get_annihilator_blockdense(basis, 1)
        c2 = get_annihilator_blockdense(basis, 2)

        hopping = c1' * c2 + c2' * c1

        @test hopping.flux.delta_qn == QN_N(0)

        # Should be Hermitian
        hopping_mat = to_matrix(hopping)
        @test hopping_mat ≈ adjoint(hopping_mat) atol=1e-14
    end

end