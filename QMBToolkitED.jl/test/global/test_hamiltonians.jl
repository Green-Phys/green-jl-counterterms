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

@testset "Hamiltonians: Non-interacting (1D Chain)" begin
    # 1D chain with L=3 sites, nearest-neighbor hopping t=2, with spin
    # Orbital ordering: (site 0 ↑, site 0 ↓, site 1 ↑, site 1 ↓, site 2 ↑, site 2 ↓)
    # Indices:          (0,      1,      2,      3,      4,      5)
    L = 3
    t = 2.0

    # Construct hopping string for 1D chain with open boundary conditions
    # Hopping between sites: 0↔1 and 1↔2, separately for each spin
    hopping_str = """
    0 2 $(-t) 0.0
    2 0 $(-t) 0.0
    1 3 $(-t) 0.0
    3 1 $(-t) 0.0
    2 4 $(-t) 0.0
    4 2 $(-t) 0.0
    3 5 $(-t) 0.0
    5 3 $(-t) 0.0
    """
    coulomb_str = "0"

    site_qns_n = [QN_N(1), QN_N(1), QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
    site_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

    # Analytical solution: diagonalize hopping matrix
    # Hopping matrix is 6×6 and block-diagonal in spin
    # Each 3×3 block is:
    # H_hop_spin = [0  -t   0 ]
    #              [-t  0  -t ]
    #              [0  -t   0 ]
    H_hop_spin = [0.0  -t   0.0;
                  -t   0.0  -t;
                  0.0  -t   0.0]
    E_spin, _ = eigen(H_hop_spin)
    # Single-particle energies: approximately [-2√2, 0, 2√2] ≈ [-2.828, 0, 2.828]

    verify_fn = (H, annh, label) -> begin
        result = solve_hamiltonian(H)
        evals = get_evals(result)
        @test length(evals) == 64  # 2^6 = 64 many-body states

        if label == "Dense"
            @test size(H) == (64, 64)
            @test ishermitian(H)
        end

        # Verify ground state energy
        # Ground state is when the two lowest single-particle states are filled
        # (one for each spin with energy E_spin[1] ≈ -2.828)
        evals_sorted = sort(evals)
        E_ground_expected = 2 * E_spin[1]  # Two fermions in lowest state
        @test evals_sorted[1] ≈ E_ground_expected atol=1e-9

        # Verify vacuum state (no particles) has energy 0
        # For 6 orbitals, vacuum is state with no particles
        # This should be one of the many-body eigenvalues
        @test any(isapprox.(evals, 0.0, atol=1e-9))

        # Verify partition function
        beta = 10.0
        Z_calculated = get_Z(evals, beta, shift_energy=true)

        # Analytical Z for non-interacting fermions (shifted)
        # For unshifted: Z = ∏_k (1 + exp(-β ε_k)) where ε_k are single-particle energies
        # For shifted: Z_shifted = exp(β E_0) * Z_unshifted
        # where E_0 is the many-body ground state energy
        E_ground = 2 * E_spin[1]  # Ground state: fill two lowest single-particle states
        Z_unshifted = prod(1 .+ exp.(-beta .* E_spin))^2
        Z_analytical = exp(beta * E_ground) * Z_unshifted
        @test Z_calculated ≈ Z_analytical atol=1e-9
    end

    run_hamiltonian_test("Non-interacting (1D Chain, L=3, t=2)",
        hopping_str=hopping_str,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn
    )
end

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
        elseif label == "BlockDense N"
             @test H.flux.delta_qn == QN_N(0)
        elseif label == "BlockDense NSz"
             @test H.flux.delta_qn == QN_NSz(0, 0)
        elseif label == "BlockDense Trivial"
             @test H.flux.delta_qn == QN_Trivial()
        end

        result = solve_hamiltonian(H)
        evals = get_evals(result)
        @test length(evals) == 64

        beta = 10.0
        Z = get_Z(evals, beta, shift_energy=true)
        @test Z ≈ 3.87936702286 atol=1e-9
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
        result = solve_hamiltonian(H)
        evals = get_evals(result)
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
        result = solve_hamiltonian(H)
        evals = get_evals(result)
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
