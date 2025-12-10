"""
Global Tests: Green's Functions

Tests Green's function calculations for both Dense and BlockDense methods.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra

@testset "Green's Functions: Non-interacting (Dense)" begin
    ϵ = 0.5
    hopping_str = "0 0 $(ϵ) 0.0"
    coulomb_str = "0"

    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
    evals, evecs = solve_hamiltonian(H)
    beta = 10.0

    # Matsubara
    n_matsubara = 100
    iwn, giwn = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)
    giwn_analytical = 1 ./ (iwn .- ϵ)
    @test giwn ≈ giwn_analytical atol=1e-9

    # Imaginary-time
    n_tau = 100
    tau_grid, gtau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1)
    Z_analytical = 1 + exp(-beta * ϵ)
    gtau_analytical = -exp.(-ϵ .* tau_grid) ./ Z_analytical
    @test gtau ≈ gtau_analytical atol=1e-9
end

@testset "Green's Functions: Non-interacting (BlockDense)" begin
    ϵ = 0.5
    hopping_str = "0 0 $(ϵ) 0.0"
    coulomb_str = "0"

    site_qns = [QN_N(1)]
    basis = make_basis(site_qns)
    H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
    evals, evecs = solve_hamiltonian(H)
    beta = 10.0

    # Matsubara
    n_matsubara = 100
    iwn, giwn = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)
    giwn_analytical = 1 ./ (iwn .- ϵ)
    @test giwn ≈ giwn_analytical atol=1e-9

    # Imaginary-time
    n_tau = 100
    tau_grid, gtau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1)
    Z_analytical = 1 + exp(-beta * ϵ)
    gtau_analytical = -exp.(-ϵ .* tau_grid) ./ Z_analytical
    @test gtau ≈ gtau_analytical atol=1e-9
end

@testset "Green's Functions: Hubbard Atom (Dense)" begin
    U = 2.0
    beta = 10.0
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

    # Matsubara
    n_matsubara = 100
    iwn, giwn = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)
    giwn_analytical = 0.5 .* (1 ./ (iwn .+ U/2) .+ 1 ./ (iwn .- U/2))
    @test giwn ≈ giwn_analytical atol=1e-9

    _, giwn_offdiag = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 2)
    @test all(isapprox.(giwn_offdiag, 0.0, atol=1e-9))

    # Imaginary-time
    n_tau = 100
    tau_grid, gtau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1)
    Z_analytical = 2 * (1 + exp(-beta * U/2))
    gtau_analytical = -(exp.(-(beta .- tau_grid) .* (U/2)) .+ exp.(-tau_grid .* (U/2))) ./ Z_analytical
    @test gtau ≈ gtau_analytical atol=1e-9

    _, gtau_offdiag = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 2)
    @test all(isapprox.(gtau_offdiag, 0.0, atol=1e-9))
end

@testset "Green's Functions: Hubbard Atom (BlockDense)" begin
    U = 2.0
    beta = 10.0
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

    # Matsubara
    n_matsubara = 100
    iwn, giwn = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1)
    giwn_analytical = 0.5 .* (1 ./ (iwn .+ U/2) .+ 1 ./ (iwn .- U/2))
    @test giwn ≈ giwn_analytical atol=1e-9

    _, giwn_offdiag = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 2)
    @test all(isapprox.(giwn_offdiag, 0.0, atol=1e-9))

    # Imaginary-time
    n_tau = 100
    tau_grid, gtau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1)
    Z_analytical = 2 * (1 + exp(-beta * U/2))
    gtau_analytical = -(exp.(-(beta .- tau_grid) .* (U/2)) .+ exp.(-tau_grid .* (U/2))) ./ Z_analytical
    @test gtau ≈ gtau_analytical atol=1e-9

    _, gtau_offdiag = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 2)
    @test all(isapprox.(gtau_offdiag, 0.0, atol=1e-9))
end
