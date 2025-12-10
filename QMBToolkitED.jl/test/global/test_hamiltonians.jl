"""
Global Tests: Hamiltonian Construction

Tests Hamiltonian construction and physical models.
These tests verify the user-facing API works correctly for both Dense and BlockDense methods.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra

@testset "Hamiltonians: Anderson Impurity (Dense)" begin
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

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    @test size(H) == (64, 64)
    @test ishermitian(H)

    evals, evecs = solve_hamiltonian(H)
    @test length(evals) == 64

    beta = 10.0
    Z = get_Z(evals, beta, shift_energy=true)
    @test Z ≈ 3.87936702286 atol=1e-6
end

@testset "Hamiltonians: Anderson Impurity (BlockDense)" begin
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

    site_qns = [QN_N(1) for _ in 1:6]
    basis = make_basis(site_qns)
    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)

    @test H.basis === basis
    @test H.flux.delta_qn == QN_N(0)

    evals, evecs = solve_hamiltonian(H)
    @test length(evals) == 64

    beta = 10.0
    Z = get_Z(evals, beta, shift_energy=true)
    @test Z ≈ 3.87936702286 atol=1e-6
end

@testset "Hamiltonians: Hubbard Atom (Dense)" begin
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

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    evals, evecs = solve_hamiltonian(H)

    @test length(evals) == 4
    @test ishermitian(H)

    beta = 10.0
    Z = get_Z(evals, beta, shift_energy=true)
    Z_analytical = 2 * (1 + exp(-beta * U/2))
    @test Z ≈ Z_analytical atol=1e-9
end

@testset "Hamiltonians: Hubbard Atom (BlockDense)" begin
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

    site_qns = [QN_N(1), QN_N(1)]
    basis = make_basis(site_qns)
    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
    evals, evecs = solve_hamiltonian(H)

    @test length(evals) == 4

    beta = 10.0
    Z = get_Z(evals, beta, shift_energy=true)
    Z_analytical = 2 * (1 + exp(-beta * U/2))
    @test Z ≈ Z_analytical atol=1e-9
end

@testset "Hamiltonians: Hubbard Dimer (Dense)" begin
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

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    evals, evecs = solve_hamiltonian(H)

    @test length(evals) == 16
    @test ishermitian(H)
end

@testset "Hamiltonians: Hubbard Dimer (BlockDense)" begin
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

    site_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
    basis = make_basis(site_qns)
    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
    evals, evecs = solve_hamiltonian(H)

    @test length(evals) == 16
end
