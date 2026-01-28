"""
Unit Tests: Basis Construction

Tests for Basis and make_basis function.
Validates basis partitioning, sector generation, and state coverage.
"""

using Test
using QMBToolkitED

@testset "Basis Construction" begin

    @testset "Simple 2-flavor spin system with QN_NSz" begin
        site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 2
        @test basis.total_dim == 4

        @test length(basis.sectors) == 4

        sector_qns = [s.qn for s in basis.sectors]
        @test QN_NSz(0, 0) in sector_qns
        @test QN_NSz(1, 1) in sector_qns
        @test QN_NSz(1, -1) in sector_qns
        @test QN_NSz(2, 0) in sector_qns

        dims = [s.dim for s in basis.sectors]
        @test sum(dims) == 4
        @test all(d == 1 for d in dims)

        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))
        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))
        @test haskey(basis.qn_to_sector, QN_NSz(1, -1))
        @test haskey(basis.qn_to_sector, QN_NSz(2, 0))

        vac_idx = basis.qn_to_sector[QN_NSz(0, 0)]
        @test basis.sectors[vac_idx].states == [0b00]

        up_idx = basis.qn_to_sector[QN_NSz(1, 1)]
        @test basis.sectors[up_idx].states == [0b01]

        down_idx = basis.qn_to_sector[QN_NSz(1, -1)]
        @test basis.sectors[down_idx].states == [0b10]

        both_idx = basis.qn_to_sector[QN_NSz(2, 0)]
        @test basis.sectors[both_idx].states == [0b11]
    end

    @testset "4-flavor Hubbard atom with QN_NSz" begin
        site_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 4
        @test basis.total_dim == 16

        total_states = sum(s.dim for s in basis.sectors)
        @test total_states == 16

        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))
        @test haskey(basis.qn_to_sector, QN_NSz(4, 0))

        vac_idx = basis.qn_to_sector[QN_NSz(0, 0)]
        @test basis.sectors[vac_idx].states == [0b0000]
        @test basis.sectors[vac_idx].dim == 1

        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))
        @test haskey(basis.qn_to_sector, QN_NSz(1, -1))

        n1_up_idx = basis.qn_to_sector[QN_NSz(1, 1)]
        @test basis.sectors[n1_up_idx].dim == 2

        n1_down_idx = basis.qn_to_sector[QN_NSz(1, -1)]
        @test basis.sectors[n1_down_idx].dim == 2

        @test haskey(basis.qn_to_sector, QN_NSz(2, 0))
        n2_sz0_idx = basis.qn_to_sector[QN_NSz(2, 0)]
        @test basis.sectors[n2_sz0_idx].dim == 4
    end

    @testset "Simple N-only system with QN_N" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 3
        @test basis.total_dim == 8
        @test length(basis.sectors) == 4

        sector_qns = [s.qn for s in basis.sectors]
        @test QN_N(0) in sector_qns
        @test QN_N(1) in sector_qns
        @test QN_N(2) in sector_qns
        @test QN_N(3) in sector_qns

        # Binomial coefficients
        @test basis.sectors[basis.qn_to_sector[QN_N(0)]].dim == 1
        @test basis.sectors[basis.qn_to_sector[QN_N(1)]].dim == 3
        @test basis.sectors[basis.qn_to_sector[QN_N(2)]].dim == 3
        @test basis.sectors[basis.qn_to_sector[QN_N(3)]].dim == 1

        @test sum(s.dim for s in basis.sectors) == 8
    end

    @testset "Larger N-only system" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 4
        @test basis.total_dim == 16
        @test length(basis.sectors) == 5

        @test basis.sectors[basis.qn_to_sector[QN_N(0)]].dim == 1
        @test basis.sectors[basis.qn_to_sector[QN_N(1)]].dim == 4
        @test basis.sectors[basis.qn_to_sector[QN_N(2)]].dim == 6
        @test basis.sectors[basis.qn_to_sector[QN_N(3)]].dim == 4
        @test basis.sectors[basis.qn_to_sector[QN_N(4)]].dim == 1
    end

    @testset "State coverage and uniqueness" begin
        for nflavors in [2, 3, 4]
            site_qns = [QN_N(1) for _ in 1:nflavors]
            basis = make_basis(site_qns)

            all_states = Int[]
            for sector in basis.sectors
                append!(all_states, sector.states)
            end

            @test length(all_states) == 1 << nflavors
            @test length(unique(all_states)) == length(all_states)
            @test all(0 <= s < (1 << nflavors) for s in all_states)
        end
    end

    @testset "State lookup in sectors" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        for sector in basis.sectors
            for (local_idx, state) in enumerate(sector.states)
                @test sector.state_to_index[state] == local_idx
                @test haskey(sector.state_to_index, state)
            end
        end

        n1_idx = basis.qn_to_sector[QN_N(1)]
        sector = basis.sectors[n1_idx]
        @test haskey(sector.state_to_index, 0b001)
        @test haskey(sector.state_to_index, 0b010)
        @test haskey(sector.state_to_index, 0b100)
    end

    @testset "Sector sorting and determinism" begin
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]

        basis1 = make_basis(site_qns)
        basis2 = make_basis(site_qns)

        @test length(basis1.sectors) == length(basis2.sectors)

        for (s1, s2) in zip(basis1.sectors, basis2.sectors)
            @test s1.qn == s2.qn
            @test s1.states == s2.states
            @test s1.dim == s2.dim
        end

        for sector in basis1.sectors
            @test issorted(sector.states)
        end
    end

    @testset "Edge cases" begin
        site_qns = [QN_N(1)]
        basis = make_basis(site_qns)
        @test basis.nflavors == 1
        @test basis.total_dim == 2
        @test length(basis.sectors) == 2

        site_qns = [QN_NSz(1, 1), QN_NSz(1, 1)]
        basis = make_basis(site_qns)
        @test basis.nflavors == 2
        @test length(basis.sectors) == 3
        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))
        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))
        @test haskey(basis.qn_to_sector, QN_NSz(2, 2))
    end

    @testset "Basis display" begin
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        io = IOBuffer()
        show(io, basis)
        output = String(take!(io))

        @test occursin("Basis", output)
        @test occursin("nflavors: 2", output)
        @test occursin("total_dim: 4", output)
    end

    @testset "Mixed site quantum numbers with QN_NSz" begin
        site_qns = [QN_NSz(1, 1), QN_NSz(1, 1), QN_NSz(1, -1)]
        basis = make_basis(site_qns)

        @test basis.nflavors == 3
        @test basis.total_dim == 8

        @test haskey(basis.qn_to_sector, QN_NSz(0, 0))
        @test haskey(basis.qn_to_sector, QN_NSz(1, 1))
        @test haskey(basis.qn_to_sector, QN_NSz(1, -1))
        @test haskey(basis.qn_to_sector, QN_NSz(2, 2))
        @test haskey(basis.qn_to_sector, QN_NSz(2, 0))
        @test haskey(basis.qn_to_sector, QN_NSz(3, 1))

        @test sum(s.dim for s in basis.sectors) == 8
    end

end
