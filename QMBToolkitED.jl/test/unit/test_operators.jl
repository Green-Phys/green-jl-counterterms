"""
Unit Tests: Operator Construction

Tests for Operator and annihilator construction.
Validates operator construction, flux calculation, and fermionic properties.
"""

using Test
using QMBToolkitED
using LinearAlgebra

@testset "Operator Construction" begin

    @testset "Simple 2-flavor operator with QN_N" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)

        @test c1.flux == QN_N(-1)
        @test c1.basis === basis
        @test !isempty(c1.blocks)

        vac_idx = basis.qn_to_sector[QN_N(0)]
        @test !any(((tgt, src),) -> src == vac_idx, keys(c1.blocks))

        c2 = annihilator(basis, 2)
        @test c2.flux == QN_N(-1)
    end

    @testset "2-flavor spin system with QN_NSz" begin
        flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = Basis(flavor_qns)

        c_up = annihilator(basis, 1)
        @test c_up.flux == QN_NSz(-1, -1)

        c_down = annihilator(basis, 2)
        @test c_down.flux == QN_NSz(-1, 1)

        @test c_up.flux != c_down.flux
    end

    @testset "Block structure for 2-flavor QN_N" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        c1 = annihilator(basis, 1)

        n0_idx = basis.qn_to_sector[QN_N(0)]
        n1_idx = basis.qn_to_sector[QN_N(1)]
        n2_idx = basis.qn_to_sector[QN_N(2)]

        @test haskey(c1.blocks, (n0_idx, n1_idx))
        @test haskey(c1.blocks, (n1_idx, n2_idx))

        block_10 = c1.blocks[(n0_idx, n1_idx)]
        @test size(block_10) == (1, 2)

        block_21 = c1.blocks[(n1_idx, n2_idx)]
        @test size(block_21) == (2, 1)
    end

    @testset "Fermionic signs in operators" begin
        flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)
        c3 = annihilator(basis, 3)

        n2_idx = basis.qn_to_sector[QN_N(2)]
        n1_idx = basis.qn_to_sector[QN_N(1)]
        n2_sector = basis.sectors[n2_idx]
        n1_sector = basis.sectors[n1_idx]

        state_110 = 0b110
        @test state_110 in n2_sector.states
        src_local = n2_sector.state_to_index[state_110]

        # c3: |110⟩ -> |010⟩, sign = -1
        block_c3 = c3.blocks[(n1_idx, n2_idx)]
        tgt_local_c3 = n1_sector.state_to_index[0b010]
        @test block_c3[tgt_local_c3, src_local] == -1.0

        state_011 = 0b011
        src_local_011 = n2_sector.state_to_index[state_011]

        # c1: |011⟩ -> |010⟩, sign = +1
        block_c1 = c1.blocks[(n1_idx, n2_idx)]
        tgt_local_c1 = n1_sector.state_to_index[0b010]
        @test block_c1[tgt_local_c1, src_local_011] == +1.0

        # c2: |011⟩ -> |001⟩, sign = -1
        block_c2 = c2.blocks[(n1_idx, n2_idx)]
        tgt_local_c2 = n1_sector.state_to_index[0b001]
        @test block_c2[tgt_local_c2, src_local_011] == -1.0
    end

    @testset "Operator action on vacuum" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        c1 = annihilator(basis, 1)

        vac_idx = basis.qn_to_sector[QN_N(0)]
        has_vac_source = any(((tgt, src),) -> src == vac_idx, keys(c1.blocks))
        @test !has_vac_source
    end

    @testset "Operator block non-zero checks" begin
        flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        c1 = annihilator(basis, 1)

        for ((tgt, src), block) in c1.blocks
            @test !iszero(block)
            @test any(!iszero, block)
        end
    end

    @testset "Multiple operators on same basis" begin
        flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        operators = [annihilator(basis, i) for i in 1:3]

        @test all(op.flux == QN_N(-1) for op in operators)
        @test all(op.basis === basis for op in operators)
        @test operators[1].blocks !== operators[2].blocks
    end

    @testset "4-flavor Hubbard atom operators" begin
        flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = Basis(flavor_qns)
        c = [annihilator(basis, i) for i in 1:4]

        @test c[1].flux == QN_NSz(-1, -1)
        @test c[2].flux == QN_NSz(-1, 1)
        @test c[3].flux == QN_NSz(-1, -1)
        @test c[4].flux == QN_NSz(-1, 1)

        @test c[1].flux == c[3].flux
        @test c[2].flux == c[4].flux
        @test all(!isempty(op.blocks) for op in c)
    end

    @testset "Operator display" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        c1 = annihilator(basis, 1)

        io = IOBuffer()
        show(io, c1)
        output = String(take!(io))

        @test occursin("Operator", output)
        @test occursin("flux", output)
        @test occursin("num_blocks", output)
        @test occursin("total_nonzeros", output)
    end

    @testset "Boundary conditions - invalid flavor" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        @test_throws AssertionError annihilator(basis, 0)
        @test_throws AssertionError annihilator(basis, 3)
    end

    @testset "Single flavor system" begin
        flavor_qns = [QN_N(1)]
        basis = Basis(flavor_qns)
        c1 = annihilator(basis, 1)

        @test c1.flux == QN_N(-1)
        @test length(c1.blocks) == 1

        n0_idx = basis.qn_to_sector[QN_N(0)]
        n1_idx = basis.qn_to_sector[QN_N(1)]

        @test haskey(c1.blocks, (n0_idx, n1_idx))
        block = c1.blocks[(n0_idx, n1_idx)]
        @test size(block) == (1, 1)
        @test block[1, 1] == 1.0
    end

    @testset "Block matrix element values" begin
        flavor_qns = [QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)
        c1 = annihilator(basis, 1)
        c2 = annihilator(basis, 2)

        n1_idx = basis.qn_to_sector[QN_N(1)]
        n2_idx = basis.qn_to_sector[QN_N(2)]

        n1_sector = basis.sectors[n1_idx]
        @test n1_sector.states == [0b01, 0b10]

        n2_sector = basis.sectors[n2_idx]
        @test n2_sector.states == [0b11]

        # c1: |11⟩ -> |10⟩, sign = +1
        block_c1 = c1.blocks[(n1_idx, n2_idx)]
        src_local = n2_sector.state_to_index[0b11]
        tgt_local = n1_sector.state_to_index[0b10]
        @test block_c1[tgt_local, src_local] == +1.0

        # c2: |11⟩ -> |01⟩, sign = -1
        block_c2 = c2.blocks[(n1_idx, n2_idx)]
        tgt_local = n1_sector.state_to_index[0b01]
        @test block_c2[tgt_local, src_local] == -1.0
    end

    @testset "All blocks have correct dimensions" begin
        flavor_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
        basis = Basis(flavor_qns)

        for flavor in 1:4
            c = annihilator(basis, flavor)
            for ((tgt_idx, src_idx), block) in c.blocks
                tgt_sector = basis.sectors[tgt_idx]
                src_sector = basis.sectors[src_idx]
                @test size(block, 1) == tgt_sector.dim
                @test size(block, 2) == src_sector.dim
                expected_tgt_qn = src_sector.qn + c.flux
                @test tgt_sector.qn == expected_tgt_qn
            end
        end
    end

    @testset "Sector connectivity from flux" begin
        flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = Basis(flavor_qns)
        c_up = annihilator(basis, 1)

        for ((tgt_idx, src_idx), block) in c_up.blocks
            tgt_qn = basis.sectors[tgt_idx].qn
            src_qn = basis.sectors[src_idx].qn
            expected_tgt = src_qn + c_up.flux
            @test tgt_qn == expected_tgt
        end
    end

end
