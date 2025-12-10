"""
Global Tests: Hamiltonian Construction

Tests Hamiltonian construction and physical models.
These tests verify the user-facing API works correctly for both Dense and BlockDense methods.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra
using .GlobalTestUtils 

@testset "Hamiltonians: Anderson Impurity" begin
    hopping_str = """
    0 0 -0.5 0.0
    1 1 -0.5 0.0
    2 2 -0.5 0.0
    3 3 -0.5 0.0
    4 4 0.5 0.0
    5 5 0.5 0.0
    2 0 1.0 0.0
    0 2 1.0 0.0
    3 1 1.0 0.0
    1 3 1.0 0.0
    4 0 1.0 0.0
    0 4 1.0 0.0
    5 1 1.0 0.0
    1 5 1.0 0.0
    """
    coulomb_str = """
    1
    0 0 1 1 0 1.0 0.0
    1 1 0 0 1 1.0 0.0
    """

    site_qns_n = [QN_N(1) for _ in 1:6]
    # Flavors [0,1] = impurity [↑,↓], [2,3] = bath1 [↑,↓], [4,5] = bath2 [↑,↓]
    site_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_fn = (H, annh, label) -> begin
        if label == "Dense"
             @test size(H) == (64, 64)
             @test ishermitian(H)
        else
             @test H.flux.delta_qn == (label == "BlockDense N" ? QN_N(0) : QN_NSz(0, 0))
        end

        evals, evecs = solve_hamiltonian(H)
        @test length(evals) == 64

        beta = 10.0
        Z = get_Z(evals, beta, shift_energy=true)
        @test Z ≈ 3.87936702286 atol=1e-6
    end

    run_hamiltonian_test("Anderson Impurity",
        hopping_str=hopping_str,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn
    )
end

@testset "Hamiltonians: Hubbard Atom" begin
    U = 2.0
    hopping_str = """
    0 0 $(-U/2) 0.0
    1 1 $(-U/2) 0.0
    """
    coulomb_str = """
    2
    0 0 1 1 0 $(U) 0.0
    1 1 0 0 1 $(U) 0.0
    """

    site_qns_n = [QN_N(1), QN_N(1)]
    site_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_fn = (H, annh, label) -> begin
        evals, evecs = solve_hamiltonian(H)
        @test length(evals) == 4
        
        if label == "Dense"
            @test ishermitian(H)
        end

        beta = 10.0
        Z = get_Z(evals, beta, shift_energy=true)
        Z_analytical = 2 * (1 + exp(-beta * U/2))
        @test Z ≈ Z_analytical atol=1e-9
    end

    run_hamiltonian_test("Hubbard Atom",
        hopping_str=hopping_str,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn
    )
end

@testset "Hamiltonians: Hubbard Dimer" begin
    t = 1.0
    U = 2.0
    hopping_str = """
    0 0 $(-U/2) 0.0
    1 1 $(-U/2) 0.0
    2 2 $(-U/2) 0.0
    3 3 $(-U/2) 0.0
    0 2 $(-t) 0.0
    2 0 $(-t) 0.0
    1 3 $(-t) 0.0
    3 1 $(-t) 0.0
    """
    coulomb_str = """
    4
    0 0 1 1 0 $(U) 0.0
    1 1 0 0 1 $(U) 0.0
    2 2 3 3 2 $(U) 0.0
    3 3 2 2 3 $(U) 0.0
    """

    site_qns_n = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
    site_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_fn = (H, annh, label) -> begin
        evals, evecs = solve_hamiltonian(H)
        @test length(evals) == 16
        
        if label == "Dense"
             @test ishermitian(H)
        end
        
        analytical_evals = [
            0.0, 0.0, 0.0,
            -U, -U, -U,
            -U/2 - t, -U/2 - t, -U/2 - t, -U/2 - t,
            -U/2 + t, -U/2 + t, -U/2 + t, -U/2 + t,
            0.5 * (-U - sqrt(16*t^2 + U^2)),
            0.5 * (-U + sqrt(16*t^2 + U^2))
        ]
        
        beta = 10.0
        Z_analytical = get_Z(analytical_evals, beta, shift_energy=true)
        Z_calculated = get_Z(evals, beta, shift_energy=true)
        @test Z_calculated ≈ Z_analytical atol=1e-9
    end

    run_hamiltonian_test("Hubbard Dimer",
        hopping_str=hopping_str,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn
    )
end
