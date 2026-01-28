"""
Unit Tests: Core Types

Tests for quantum number types, sector construction, and utility functions.
"""

using Test
using QMBToolkitED

@testset "Core Types" begin

    @testset "QN_NSz Type" begin
        @testset "Construction and Equality" begin
            qn1 = QN_NSz(1, 1)
            qn2 = QN_NSz(1, 1)
            qn3 = QN_NSz(1, -1)
            qn4 = QN_NSz(2, 0)

            @test qn1 == qn2
            @test qn1 != qn3
            @test qn1 != qn4
        end

        @testset "Hashing (for Dict usage)" begin
            qn1 = QN_NSz(1, 1)
            qn2 = QN_NSz(1, 1)
            qn3 = QN_NSz(1, -1)

            # Same QN should hash to same value
            @test hash(qn1) == hash(qn2)
            # Different QN should (probably) hash differently
            @test hash(qn1) != hash(qn3)

            # Should work in Dict
            d = Dict(qn1 => "up", qn3 => "down")
            @test d[qn2] == "up"  # qn2 == qn1
        end

        @testset "Arithmetic: +" begin
            qn_up = QN_NSz(1, 1)
            qn_down = QN_NSz(1, -1)

            result = qn_up + qn_down
            @test result == QN_NSz(2, 0)

            # Add zero
            qn_zero = QN_NSz(0, 0)
            @test qn_up + qn_zero == qn_up
        end

        @testset "Arithmetic: -" begin
            qn = QN_NSz(1, 1)
            rev = -qn
            @test rev == QN_NSz(-1, -1)

            # Double negate
            @test -rev == qn

            # Negate zero
            @test -QN_NSz(0, 0) == QN_NSz(0, 0)
        end

        @testset "Arithmetic: zero" begin
            z = zero(QN_NSz)
            @test z == QN_NSz(0, 0)

            # Zero is identity for +
            qn = QN_NSz(2, -1)
            @test qn + z == qn
            @test z + qn == qn
        end

        @testset "Ordering" begin
            qn1 = QN_NSz(0, 0)
            qn2 = QN_NSz(1, -1)
            qn3 = QN_NSz(1, 0)
            qn4 = QN_NSz(1, 1)
            qn5 = QN_NSz(2, 0)

            # Sort by n first, then sz
            @test qn1 < qn2
            @test qn2 < qn3
            @test qn3 < qn4
            @test qn4 < qn5

            # Sorting should work
            unsorted = [qn4, qn1, qn5, qn2, qn3]
            sorted_qns = sort(unsorted)
            @test sorted_qns == [qn1, qn2, qn3, qn4, qn5]
        end

        @testset "Display" begin
            qn = QN_NSz(2, -1)
            str = sprint(show, qn)
            @test occursin("QN_NSz", str)
            @test occursin("n=2", str)
            @test occursin("sz=-1", str)
        end
    end

    @testset "QN_N Type" begin
        @testset "Construction and Equality" begin
            qn1 = QN_N(1)
            qn2 = QN_N(1)
            qn3 = QN_N(2)

            @test qn1 == qn2
            @test qn1 != qn3
        end

        @testset "Hashing" begin
            qn1 = QN_N(1)
            qn2 = QN_N(1)
            qn3 = QN_N(2)

            @test hash(qn1) == hash(qn2)

            # Dict usage
            d = Dict(qn1 => "one", qn3 => "two")
            @test d[qn2] == "one"
        end

        @testset "Arithmetic: +" begin
            qn1 = QN_N(1)
            qn2 = QN_N(2)

            result = qn1 + qn2
            @test result == QN_N(3)

            # Add zero
            qn_zero = QN_N(0)
            @test qn1 + qn_zero == qn1
        end

        @testset "Arithmetic: -" begin
            qn = QN_N(3)
            rev = -qn
            @test rev == QN_N(-3)

            # Double negate
            @test -rev == qn
        end

        @testset "Arithmetic: zero" begin
            z = zero(QN_N)
            @test z == QN_N(0)

            # Identity
            qn = QN_N(5)
            @test qn + z == qn
        end

        @testset "Ordering" begin
            qn1 = QN_N(0)
            qn2 = QN_N(1)
            qn3 = QN_N(2)

            @test qn1 < qn2 < qn3

            unsorted = [qn3, qn1, qn2]
            @test sort(unsorted) == [qn1, qn2, qn3]
        end

        @testset "Display" begin
            qn = QN_N(5)
            str = sprint(show, qn)
            @test occursin("QN_N", str)
            @test occursin("n=5", str)
        end
    end

    @testset "Sector Type" begin
        @testset "Construction with validation" begin
            qn = QN_N(1)
            states = [1, 2, 4]  # Binary: 001, 010, 100
            state_to_index = Dict(1=>1, 2=>2, 4=>3)
            dim = 3

            sector = Sector(qn, states, state_to_index, dim)

            @test sector.qn == qn
            @test sector.states == states
            @test sector.state_to_index == state_to_index
            @test sector.dim == 3
        end

        @testset "Convenience constructor" begin
            qn = QN_N(2)
            states = [3, 5, 6]  # Binary: 011, 101, 110

            sector = Sector(qn, states)

            @test sector.qn == qn
            @test sector.states == states
            @test sector.dim == 3
            @test sector.state_to_index[3] == 1
            @test sector.state_to_index[5] == 2
            @test sector.state_to_index[6] == 3
        end

        @testset "Dimension mismatch error" begin
            qn = QN_N(1)
            states = [1, 2]
            state_to_index = Dict(1=>1, 2=>2)

            # Wrong dim
            @test_throws AssertionError Sector(qn, states, state_to_index, 3)
        end

        @testset "State lookup" begin
            qn = QN_NSz(1, 1)
            states = [1, 4]  # Bit 0 and bit 2 occupied
            sector = Sector(qn, states)

            # Lookup should be O(1)
            @test sector.state_to_index[1] == 1
            @test sector.state_to_index[4] == 2

            # Access state from index
            @test sector.states[1] == 1
            @test sector.states[2] == 4
        end

        @testset "Display" begin
            qn = QN_N(2)
            states = [3, 5, 6]
            sector = Sector(qn, states)

            str = sprint(show, sector)
            @test occursin("Sector", str)
            @test occursin("dim=3", str)
        end
    end

    @testset "total_qn" begin
        @testset "QN_NSz: Simple 2-flavor spin system" begin
            # [↑, ↓]
            flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]

            # State |00⟩ = 0b00 = 0: vacuum
            @test total_qn(0, flavor_qns) == QN_NSz(0, 0)

            # State |01⟩ = 0b01 = 1: bit 0 set → flavor 1 (↑) occupied
            @test total_qn(1, flavor_qns) == QN_NSz(1, 1)

            # State |10⟩ = 0b10 = 2: bit 1 set → flavor 2 (↓) occupied
            @test total_qn(2, flavor_qns) == QN_NSz(1, -1)

            # State |11⟩ = 0b11 = 3: both occupied
            @test total_qn(3, flavor_qns) == QN_NSz(2, 0)
        end

        @testset "QN_NSz: 4-flavor Hubbard atom" begin
            # [↑, ↓, ↑, ↓] (flavors 1-2 for first atom, flavors 3-4 for second atom)
            flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

            # State |0000⟩ = 0b0000 = 0
            @test total_qn(0, flavor_qns) == QN_NSz(0, 0)

            # State |0101⟩ = 0b0101 = 5: bits 0,2 set → flavors 1,3 (both ↑)
            @test total_qn(5, flavor_qns) == QN_NSz(2, 2)

            # State |1010⟩ = 0b1010 = 10: bits 1,3 set → flavors 2,4 (both ↓)
            @test total_qn(10, flavor_qns) == QN_NSz(2, -2)

            # State |0011⟩ = 0b0011 = 3: bits 0,1 set → flavors 1,2 (first atom doubly occupied)
            @test total_qn(3, flavor_qns) == QN_NSz(2, 0)

            # State |1111⟩ = 0b1111 = 15: all occupied
            @test total_qn(15, flavor_qns) == QN_NSz(4, 0)
        end

        @testset "QN_N: N-only conservation" begin
            # All flavors have QN_N(1)
            flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]

            # State |000⟩ = 0b000 = 0
            @test total_qn(0, flavor_qns) == QN_N(0)

            # State |001⟩ = 0b001 = 1
            @test total_qn(1, flavor_qns) == QN_N(1)

            # State |011⟩ = 0b011 = 3
            @test total_qn(3, flavor_qns) == QN_N(2)

            # State |111⟩ = 0b111 = 7
            @test total_qn(7, flavor_qns) == QN_N(3)
        end
    end

    @testset "fermion_sign" begin
        @testset "Basic sign calculation" begin
            @test fermion_sign(0, 1) == 1
            @test fermion_sign(1, 1) == 1
            @test fermion_sign(1, 2) == -1
            @test fermion_sign(1, 3) == -1
        end

        @testset "Multiple occupied flavors" begin
            state = 7  # |0111⟩ = 0b0111: bits 0,1,2 set
            @test fermion_sign(state, 1) == 1
            @test fermion_sign(state, 2) == -1
            @test fermion_sign(state, 3) == 1
            @test fermion_sign(state, 4) == -1
        end

        @testset "Low occupation" begin
            state = 10  # |1010⟩ = 0b1010: bits 1,3 set
            @test fermion_sign(state, 1) == 1
            @test fermion_sign(state, 2) == 1
            @test fermion_sign(state, 3) == -1
            @test fermion_sign(state, 4) == -1
            @test fermion_sign(state, 5) == 1
        end

        @testset "Consistency with Jordan-Wigner" begin
            state = 5  # |0101⟩: bits 0,2 set
            @test fermion_sign(state, 3) == -1
            @test fermion_sign(state, 1) == 1
        end
    end

end
