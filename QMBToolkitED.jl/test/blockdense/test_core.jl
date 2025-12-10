"""
BlockDense Tests: Core Functionality

Tests BlockDense-specific implementation details:
- Quantum number arithmetic
- Sector construction from site_qns
- Operator flux computation
"""

using Test
using Main.BlockDense

@testset "BlockDense: Core Functionality" begin

    @testset "Quantum Number Arithmetic" begin
        # QN_N operations
        qn1 = QN_N(1)
        @test combine_qn(qn1, qn1) == QN_N(2)
        @test reverse_qn(qn1) == QN_N(-1)
        @test zero_qn(QN_N) == QN_N(0)

        # QN_NSz operations
        qn_up = QN_NSz(1, 1)
        qn_down = QN_NSz(1, -1)
        @test combine_qn(qn_up, qn_down) == QN_NSz(2, 0)
        @test reverse_qn(qn_up) == QN_NSz(-1, -1)
        @test zero_qn(QN_NSz) == QN_NSz(0, 0)
    end

    @testset "Sector Partitioning" begin
        # Basis partitions Fock space into sectors
        site_qns = [QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)

        # All 2^nflavors states accounted for
        @test basis.total_dim == 8
        @test sum(sec.dim for sec in basis.sectors) == 8

        # No duplicate states across sectors
        all_states = Int[]
        for sec in basis.sectors
            append!(all_states, sec.states)
        end
        @test length(unique(all_states)) == length(all_states)
    end

    @testset "Operator Flux from site_qns" begin
        # Flux determined by site_qns
        site_qns_n = [QN_N(1), QN_N(1)]
        basis_n = make_basis(site_qns_n)
        c_n = get_annihilator_blockdense(basis_n, 1)
        @test c_n.flux.delta_qn == QN_N(-1)

        # Different QN type gives different flux
        site_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1)]
        basis_sz = make_basis(site_qns_sz)
        c_up = get_annihilator_blockdense(basis_sz, 1)  # Remove spin-up
        c_down = get_annihilator_blockdense(basis_sz, 2)  # Remove spin-down
        @test c_up.flux.delta_qn == QN_NSz(-1, -1)
        @test c_down.flux.delta_qn == QN_NSz(-1, 1)
    end

end
