"""
Phase 2 Tests: Site-Based Basis Construction

Tests for BlockSparseBasis and make_basis function.
Validates basis partitioning, sector generation, and state coverage.
"""

using Test
using Main.BlockSparse

@testset "Phase 2: Basis Construction" begin

    @testset "Simple 2-flavor spin system with QN_NSz" begin
        # [↑, ↓] system
        site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        # Basic properties
        @test basis.nflavors == 2
        @test basis.total_dim == 4  # 2^2 = 4 states

        # Should have 4 sectors: (N=0, Sz=0), (N=1, Sz=1), (N=1, Sz=-1), (N=2, Sz=0)
        @test length(basis.sectors) == 4

        # Check sector quantum numbers
        sector_qns = [s.qn for s in basis.sectors]
        @test QN_NSz(0, 0) in sector_qns   # Vacuum
        @test QN_NSz(1, 1) in sector_qns    # ↑ only
        @test QN_NSz(1, -1) in sector_qns   # ↓ only
        @test QN_NSz(2, 0) in sector_qns    # Both

        # Check dimensions
        dims = [s.dim for s in basis.sectors]
        @test sum(dims) == 4  # All states accounted for
        @test all(d == 1 for d in dims)  # Each sector has 1 state

        # Check qn_to_sector mapping
        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))
        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))
        @test haskey(basis.qn_to_sector, QN_NSz(1, -1))
        @test haskey(basis.qn_to_sector, QN_NSz(2, 0))

        # Verify states
        vac_idx = basis.qn_to_sector[QN_NSz(0, 0)]
        @test basis.sectors[vac_idx].states == [0b00]  # |00⟩

        up_idx = basis.qn_to_sector[QN_NSz(1, 1)]
        @test basis.sectors[up_idx].states == [0b01]  # |01⟩ = spin up (bit 0 set → flavor 1)

        down_idx = basis.qn_to_sector[QN_NSz(1, -1)]
        @test basis.sectors[down_idx].states == [0b10]  # |10⟩ = spin down (bit 1 set → flavor 2)

        both_idx = basis.qn_to_sector[QN_NSz(2, 0)]
        @test basis.sectors[both_idx].states == [0b11]  # |11⟩
    end

    @testset "4-flavor Hubbard atom with QN_NSz" begin
        # [↑, ↓, ↑, ↓] - two sites, two spins each
        site_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 4
        @test basis.total_dim == 16  # 2^4 = 16 states

        # Check all states covered
        total_states = sum(s.dim for s in basis.sectors)
        @test total_states == 16

        # Check vacuum and fully occupied
        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))   # Vacuum
        @test haskey(basis.qn_to_sector, QN_NSz(4, 0))   # All occupied

        vac_idx = basis.qn_to_sector[QN_NSz(0, 0)]
        @test basis.sectors[vac_idx].states == [0b0000]
        @test basis.sectors[vac_idx].dim == 1

        # Check N=1 sectors
        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))   # One ↑
        @test haskey(basis.qn_to_sector, QN_NSz(1, -1))  # One ↓

        n1_up_idx = basis.qn_to_sector[QN_NSz(1, 1)]
        @test basis.sectors[n1_up_idx].dim == 2  # Two ↑ orbitals

        n1_down_idx = basis.qn_to_sector[QN_NSz(1, -1)]
        @test basis.sectors[n1_down_idx].dim == 2  # Two ↓ orbitals

        # Check N=2, Sz=0 sector (largest)
        @test haskey(basis.qn_to_sector, QN_NSz(2, 0))
        n2_sz0_idx = basis.qn_to_sector[QN_NSz(2, 0)]
        # States with N=2, Sz=0: one ↑ and one ↓ electron
        # Possible: |0011⟩ (site 1 doubly occ), |0110⟩ (↓ on 1, ↑ on 2), |1001⟩ (↑ on 1, ↓ on 2), |1100⟩ (site 2 doubly occ)
        @test basis.sectors[n2_sz0_idx].dim == 4
    end

    @testset "Simple N-only system with QN_N" begin
        # 3 orbitals, N conservation only
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 3
        @test basis.total_dim == 8  # 2^3 = 8 states

        # Should have 4 sectors: N=0, 1, 2, 3
        @test length(basis.sectors) == 4

        # Check sector quantum numbers
        sector_qns = [s.qn for s in basis.sectors]
        @test QN_N(0) in sector_qns
        @test QN_N(1) in sector_qns
        @test QN_N(2) in sector_qns
        @test QN_N(3) in sector_qns

        # Check dimensions (binomial coefficients)
        n0_idx = basis.qn_to_sector[QN_N(0)]
        @test basis.sectors[n0_idx].dim == 1  # C(3,0) = 1

        n1_idx = basis.qn_to_sector[QN_N(1)]
        @test basis.sectors[n1_idx].dim == 3  # C(3,1) = 3

        n2_idx = basis.qn_to_sector[QN_N(2)]
        @test basis.sectors[n2_idx].dim == 3  # C(3,2) = 3

        n3_idx = basis.qn_to_sector[QN_N(3)]
        @test basis.sectors[n3_idx].dim == 1  # C(3,3) = 1

        # Total states
        @test sum(s.dim for s in basis.sectors) == 8
    end

    @testset "Larger N-only system" begin
        # 4 orbitals
        site_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 4
        @test basis.total_dim == 16
        @test length(basis.sectors) == 5  # N=0,1,2,3,4

        # Check binomial coefficients
        @test basis.sectors[basis.qn_to_sector[QN_N(0)]].dim == 1   # C(4,0)
        @test basis.sectors[basis.qn_to_sector[QN_N(1)]].dim == 4   # C(4,1)
        @test basis.sectors[basis.qn_to_sector[QN_N(2)]].dim == 6   # C(4,2)
        @test basis.sectors[basis.qn_to_sector[QN_N(3)]].dim == 4   # C(4,3)
        @test basis.sectors[basis.qn_to_sector[QN_N(4)]].dim == 1   # C(4,4)
    end

    @testset "State coverage and uniqueness" begin
        # Every Fock state should appear exactly once
        for nflavors in [2, 3, 4]
            site_qns = [QN_N(1) for _ in 1:nflavors]
            basis = make_basis(site_qns)

            # Collect all states from all sectors
            all_states = Int[]
            for sector in basis.sectors
                append!(all_states, sector.states)
            end

            # Check: all states covered
            @test length(all_states) == 1 << nflavors

            # Check: no duplicates
            @test length(unique(all_states)) == length(all_states)

            # Check: all states in range [0, 2^nflavors - 1]
            @test all(0 <= s < (1 << nflavors) for s in all_states)
        end
    end

    @testset "State lookup in sectors" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        # Check that state_to_index works correctly
        for sector in basis.sectors
            for (local_idx, state) in enumerate(sector.states)
                @test sector.state_to_index[state] == local_idx
                @test haskey(sector.state_to_index, state)
            end
        end

        # Check lookup performance (should be O(1))
        n1_idx = basis.qn_to_sector[QN_N(1)]
        sector = basis.sectors[n1_idx]

        # States with N=1: |001⟩=0b001=1, |010⟩=0b010=2, |100⟩=0b100=4
        @test haskey(sector.state_to_index, 0b001)
        @test haskey(sector.state_to_index, 0b010)
        @test haskey(sector.state_to_index, 0b100)
    end

    @testset "Sector sorting and determinism" begin
        # Sectors should be in deterministic order
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]

        basis1 = make_basis(site_qns)
        basis2 = make_basis(site_qns)

        # Same number of sectors
        @test length(basis1.sectors) == length(basis2.sectors)

        # Same order
        for (s1, s2) in zip(basis1.sectors, basis2.sectors)
            @test s1.qn == s2.qn
            @test s1.states == s2.states
            @test s1.dim == s2.dim
        end

        # Check states within sectors are sorted
        for sector in basis1.sectors
            @test issorted(sector.states)
        end
    end

    @testset "Edge cases" begin
        # Single orbital
        site_qns = [QN_N(1)]
        basis = make_basis(site_qns)
        @test basis.nflavors == 1
        @test basis.total_dim == 2
        @test length(basis.sectors) == 2  # N=0, N=1

        # All same spin
        site_qns = [QN_NSz(1, 1), QN_NSz(1, 1)]
        basis = make_basis(site_qns)
        @test basis.nflavors == 2
        # Sectors: (0,0), (1,1), (2,2)
        @test length(basis.sectors) == 3
        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))
        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))
        @test haskey(basis.qn_to_sector, QN_NSz(2, 2))
    end

    @testset "Basis display" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        # Should not error
        io = IOBuffer()
        show(io, basis)
        output = String(take!(io))

        @test occursin("BlockSparseBasis", output)
        @test occursin("nflavors: 2", output)
        @test occursin("total_dim: 4", output)
    end

    @testset "Mixed site quantum numbers with QN_NSz" begin
        # Different spin configurations on different orbitals
        site_qns = [QN_NSz(1, 1), QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 3
        @test basis.total_dim == 8

        # Check some specific sectors exist
        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))   # Vacuum
        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))   # One ↑
        @test haskey(basis.qn_to_sector, QN_NSz(1, -1))  # One ↓
        @test haskey(basis.qn_to_sector, QN_NSz(2, 2))   # Two ↑
        @test haskey(basis.qn_to_sector, QN_NSz(2, 0))   # ↑ + ↓
        @test haskey(basis.qn_to_sector, QN_NSz(3, 1))   # All three

        # All states accounted for
        @test sum(s.dim for s in basis.sectors) == 8
    end

end

println("\nPhase 2: Basis Construction Tests Complete")
