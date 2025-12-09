"""
Phase 2 Tests: Block-Sparse Operators

Tests for BlockSparseOperator, OperatorFlux, and get_annihilator_blocksparse.
Validates operator construction, flux calculation, and fermionic properties.
"""

using Test
using Main.BlockSparse
using LinearAlgebra

@testset "Phase 2: Operator Construction" begin

    @testset "OperatorFlux type" begin
        # Test flux construction and display
        flux1 = OperatorFlux(QN_N(-1))
        @test flux1.delta_qn == QN_N(-1)

        flux2 = OperatorFlux(QN_NSz(-1, -1))
        @test flux2.delta_qn == QN_NSz(-1, -1)

        # Display should not error
        io = IOBuffer()
        show(io, flux1)
        output = String(take!(io))
        @test occursin("OperatorFlux", output)
    end

    @testset "Simple 2-flavor operator with QN_N" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        # Create annihilator for orbital 1
        c1 = get_annihilator_blocksparse(basis, 1)

        # Check flux
        @test c1.flux.delta_qn == QN_N(-1)
        @test c1.basis === basis

        # Check blocks exist
        @test !isempty(c1.blocks)

        # Annihilating from vacuum should give nothing
        # Vacuum is in N=0 sector, target would be N=-1 (doesn't exist)
        vac_idx = basis.qn_to_sector[QN_N(0)]
        # No block should connect from N=0
        @test !any(((tgt, src),) -> src == vac_idx, keys(c1.blocks))

        # Create annihilator for orbital 2
        c2 = get_annihilator_blocksparse(basis, 2)
        @test c2.flux.delta_qn == QN_N(-1)
    end

    @testset "2-flavor spin system with QN_NSz" begin
        site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]  # [↑, ↓]
        basis = make_basis(site_qns)

        # Annihilate spin-up
        c_up = get_annihilator_blocksparse(basis, 1)
        @test c_up.flux.delta_qn == QN_NSz(-1, -1)  # Removes (N=1, Sz=1)

        # Annihilate spin-down
        c_down = get_annihilator_blocksparse(basis, 2)
        @test c_down.flux.delta_qn == QN_NSz(-1, 1)  # Removes (N=1, Sz=-1)

        # Different fluxes for different spins
        @test c_up.flux.delta_qn != c_down.flux.delta_qn
    end

    @testset "Block structure for 2-flavor QN_N" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        c1 = get_annihilator_blocksparse(basis, 1)

        # Sectors: N=0 (vacuum), N=1 (single), N=2 (double)
        n0_idx = basis.qn_to_sector[QN_N(0)]
        n1_idx = basis.qn_to_sector[QN_N(1)]
        n2_idx = basis.qn_to_sector[QN_N(2)]

        # c1 should connect N=1 -> N=0 and N=2 -> N=1
        @test haskey(c1.blocks, (n0_idx, n1_idx))
        @test haskey(c1.blocks, (n1_idx, n2_idx))

        # Check block dimensions
        block_10 = c1.blocks[(n0_idx, n1_idx)]
        @test size(block_10) == (1, 2)  # Target N=0 (dim=1), source N=1 (dim=2)

        block_21 = c1.blocks[(n1_idx, n2_idx)]
        @test size(block_21) == (2, 1)  # Target N=1 (dim=2), source N=2 (dim=1)
    end

    @testset "Fermionic signs in operators" begin
        # Test that signs are correct
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blocksparse(basis, 1)
        c2 = get_annihilator_blocksparse(basis, 2)
        c3 = get_annihilator_blocksparse(basis, 3)

        # Convention: state = bit pattern directly, |abc⟩ = 0babc (binary literal)
        # State |110⟩ = 0b110 = 6 (decimal), bits 1 and 2 set → orbitals 1,2 occupied → flavors 2,3 occupied
        # c1 annihilates bit 0 (orbital 0, flavor 1): state |110⟩ remains |110⟩ (bit 0 not set, no action)
        # c2 annihilates bit 1 (orbital 1, flavor 2): state |110⟩ → |100⟩ (bit 1 removed)
        # c3 annihilates bit 2 (orbital 2, flavor 3): state |110⟩ → |010⟩ (bit 2 removed)

        # Find sectors
        n2_idx = basis.qn_to_sector[QN_N(2)]
        n1_idx = basis.qn_to_sector[QN_N(1)]

        n2_sector = basis.sectors[n2_idx]
        n1_sector = basis.sectors[n1_idx]

        # State |110⟩ = 0b110 = 6 (bits 1,2 set → flavors 2,3)
        state_110 = 0b110
        @test state_110 in n2_sector.states
        src_local = n2_sector.state_to_index[state_110]

        # c3 annihilates flavor 3 (bit 2): |110⟩ -> |010⟩
        # Bits below bit 2: bits 0,1 with bit 1 set → 1 occupied below → sign = -1
        block_c3 = c3.blocks[(n1_idx, n2_idx)]
        state_010 = 0b010
        tgt_local_c3 = n1_sector.state_to_index[state_010]
        @test block_c3[tgt_local_c3, src_local] == -1.0

        # State |011⟩ = 0b011 = 3 (bits 0,1 set → flavors 1,2)
        state_011 = 0b011
        @test state_011 in n2_sector.states
        src_local_011 = n2_sector.state_to_index[state_011]

        # c1 annihilates flavor 1 (bit 0): |011⟩ -> |010⟩
        # No occupied bits below bit 0, so sign is +1
        block_c1 = c1.blocks[(n1_idx, n2_idx)]
        state_010_from_c1 = 0b010
        tgt_local_c1 = n1_sector.state_to_index[state_010_from_c1]
        @test block_c1[tgt_local_c1, src_local_011] == +1.0

        # c2 annihilates flavor 2 (bit 1): |011⟩ -> |001⟩
        # One occupied bit below bit 1 (bit 0 is set), so sign is -1
        block_c2 = c2.blocks[(n1_idx, n2_idx)]
        state_001 = 0b001
        tgt_local_c2 = n1_sector.state_to_index[state_001]
        @test block_c2[tgt_local_c2, src_local_011] == -1.0
    end

    @testset "Operator action on vacuum" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blocksparse(basis, 1)

        # Vacuum sector
        vac_idx = basis.qn_to_sector[QN_N(0)]

        # No blocks should have vacuum as source (can't annihilate from vacuum)
        has_vac_source = any(((tgt, src),) -> src == vac_idx, keys(c1.blocks))
        @test !has_vac_source
    end

    @testset "Operator block non-zero checks" begin
        # Blocks should only be stored if non-zero
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blocksparse(basis, 1)

        # All stored blocks should be non-zero
        for ((tgt, src), block) in c1.blocks
            @test !iszero(block)
            @test any(!iszero, block)
        end
    end

    @testset "Multiple operators on same basis" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        operators = [get_annihilator_blocksparse(basis, i) for i in 1:3]

        # All should have same flux
        @test all(op.flux.delta_qn == QN_N(-1) for op in operators)

        # All should reference same basis
        @test all(op.basis === basis for op in operators)

        # Different operators should have different blocks
        @test operators[1].blocks !== operators[2].blocks
        @test operators[2].blocks !== operators[3].blocks
    end

    @testset "4-flavor Hubbard atom operators" begin
        site_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        # Create all annihilators
        c = [get_annihilator_blocksparse(basis, i) for i in 1:4]

        # Check fluxes
        @test c[1].flux.delta_qn == QN_NSz(-1, -1)  # ↑ on site 1
        @test c[2].flux.delta_qn == QN_NSz(-1, 1)   # ↓ on site 1
        @test c[3].flux.delta_qn == QN_NSz(-1, -1)  # ↑ on site 2
        @test c[4].flux.delta_qn == QN_NSz(-1, 1)   # ↓ on site 2

        # Operators for same spin should have same flux
        @test c[1].flux.delta_qn == c[3].flux.delta_qn
        @test c[2].flux.delta_qn == c[4].flux.delta_qn

        # All should have blocks
        @test all(!isempty(op.blocks) for op in c)
    end

    @testset "Operator display" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        c1 = get_annihilator_blocksparse(basis, 1)

        # Should not error
        io = IOBuffer()
        show(io, c1)
        output = String(take!(io))

        @test occursin("BlockSparseOperator", output)
        @test occursin("flux", output)
        @test occursin("num_blocks", output)
        @test occursin("total_nonzeros", output)
    end

    @testset "Nonzero count" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        c1 = get_annihilator_blocksparse(basis, 1)

        # Count total nonzeros
        total_nnz = sum(length(block) for block in values(c1.blocks))
        @test total_nnz > 0

        # For c1 on 3 orbitals with N conservation:
        # Should have nonzeros connecting occupied states
        # Total should be sum of block dimensions
        expected_nnz = sum(prod(size(block)) for block in values(c1.blocks))
        @test total_nnz == expected_nnz
    end

    @testset "Boundary conditions - invalid flavor" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        # Should error for out-of-range flavor
        @test_throws AssertionError get_annihilator_blocksparse(basis, 0)
        @test_throws AssertionError get_annihilator_blocksparse(basis, 3)
    end

    @testset "Single flavor system" begin
        site_qns = [QN_N(1)]
        basis = make_basis(site_qns)

        c1 = get_annihilator_blocksparse(basis, 1)

        @test c1.flux.delta_qn == QN_N(-1)
        @test length(c1.blocks) == 1  # Only N=1 -> N=0

        # Check the single block
        n0_idx = basis.qn_to_sector[QN_N(0)]
        n1_idx = basis.qn_to_sector[QN_N(1)]

        @test haskey(c1.blocks, (n0_idx, n1_idx))
        block = c1.blocks[(n0_idx, n1_idx)]
        @test size(block) == (1, 1)
        @test block[1, 1] == 1.0  # No fermion sign for first orbital
    end

    @testset "Block matrix element values" begin
        # Verify specific matrix elements match expected values
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        c1 = get_annihilator_blocksparse(basis, 1)
        c2 = get_annihilator_blocksparse(basis, 2)

        n1_idx = basis.qn_to_sector[QN_N(1)]
        n2_idx = basis.qn_to_sector[QN_N(2)]

        # Convention: |ab⟩ = 0bab (binary literal). State |10⟩ = 0b10 = 2 (decimal), |01⟩ = 0b01 = 1 (decimal)
        # N=1 sector has states (sorted): |01⟩ = 0b01 = 1, |10⟩ = 0b10 = 2
        n1_sector = basis.sectors[n1_idx]
        @test n1_sector.states == [0b01, 0b10]

        # N=2 sector has state: |11⟩ = 0b11 = 3
        n2_sector = basis.sectors[n2_idx]
        @test n2_sector.states == [0b11]

        # c1 annihilates flavor 1 (bit 0): |11⟩ -> |10⟩
        # No occupied bits below bit 0, so sign is +1
        block_c1 = c1.blocks[(n1_idx, n2_idx)]
        state_11 = 0b11
        state_10 = 0b10  # |10⟩ = 0b10 = 2 (bit 1 set → flavor 2)
        src_local = n2_sector.state_to_index[state_11]
        tgt_local = n1_sector.state_to_index[state_10]
        @test block_c1[tgt_local, src_local] == +1.0

        # c2 annihilates flavor 2 (bit 1): |11⟩ -> |01⟩
        # One occupied bit below bit 1 (bit 0 is set), so sign is -1
        block_c2 = c2.blocks[(n1_idx, n2_idx)]
        state_01 = 0b01  # |01⟩ = 0b01 = 1 (bit 0 set → flavor 1)
        tgt_local = n1_sector.state_to_index[state_01]
        @test block_c2[tgt_local, src_local] == -1.0
    end

    @testset "All blocks have correct dimensions" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        for flavor in 1:4
            c = get_annihilator_blocksparse(basis, flavor)

            for ((tgt_idx, src_idx), block) in c.blocks
                tgt_sector = basis.sectors[tgt_idx]
                src_sector = basis.sectors[src_idx]

                # Block should have dimensions matching sectors
                @test size(block, 1) == tgt_sector.dim
                @test size(block, 2) == src_sector.dim

                # Target QN should equal source QN + flux
                expected_tgt_qn = combine_qn(src_sector.qn, c.flux.delta_qn)
                @test tgt_sector.qn == expected_tgt_qn
            end
        end
    end

    @testset "Sector connectivity from flux" begin
        site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        c_up = get_annihilator_blocksparse(basis, 1)  # Flux: (-1, -1)

        # Should only connect sectors where target = source + flux
        for ((tgt_idx, src_idx), block) in c_up.blocks
            tgt_qn = basis.sectors[tgt_idx].qn
            src_qn = basis.sectors[src_idx].qn
            expected_tgt = combine_qn(src_qn, c_up.flux.delta_qn)

            @test tgt_qn == expected_tgt
        end
    end

end

println("\nPhase 2: Operator Construction Tests Complete")
