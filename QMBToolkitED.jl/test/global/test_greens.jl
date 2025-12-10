"""
Global Tests: Green's Functions

Tests Green's function calculations for both Dense and BlockDense methods.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra
using .GlobalTestUtils

@testset "Green's Functions: Non-interacting" begin
    ϵ = 0.5
    hopping_str = "0 0 $(ϵ) 0.0"
    coulomb_str = "0"

    site_qns_n = [QN_N(1)]    
    site_qns_sz = [QN_NSz(1, 1)] 
    
    verify_fn = (result, annh) -> begin
        beta = 10.0
        
        # Matsubara
        n_matsubara = 100
        iwn, giwn = calculate_green_matsubara(result, annh, beta, n_matsubara, 1, 1)
        giwn_analytical = 1 ./ (iwn .- ϵ)
        @test giwn ≈ giwn_analytical atol=1e-9

        # Imaginary-time
        n_tau = 100
        tau_grid, gtau = calculate_green_tau(result, annh, beta, n_tau, 1, 1)
        Z_analytical = 1 + exp(-beta * ϵ)
        gtau_analytical = -exp.(-ϵ .* tau_grid) ./ Z_analytical
        @test gtau ≈ gtau_analytical atol=1e-9
    end

    run_greens_test("Non-interacting",
        hopping_str=hopping_str,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn
    )
end

@testset "Green's Functions: Hubbard Atom" begin
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
    
    site_qns_n = [QN_N(1), QN_N(1)]
    site_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_fn = (result, annh) -> begin
        # Matsubara
        n_matsubara = 100
        iwn, giwn = calculate_green_matsubara(result, annh, beta, n_matsubara, 1, 1)
        giwn_analytical = 0.5 .* (1 ./ (iwn .+ U/2) .+ 1 ./ (iwn .- U/2))
        @test giwn ≈ giwn_analytical atol=1e-9

        _, giwn_offdiag = calculate_green_matsubara(result, annh, beta, n_matsubara, 1, 2)
        @test all(isapprox.(giwn_offdiag, 0.0, atol=1e-9))

        # Imaginary-time
        n_tau = 100
        tau_grid, gtau = calculate_green_tau(result, annh, beta, n_tau, 1, 1)
        Z_analytical = 2 * (1 + exp(-beta * U/2))
        gtau_analytical = -(exp.(-(beta .- tau_grid) .* (U/2)) .+ exp.(-tau_grid .* (U/2))) ./ Z_analytical
        @test gtau ≈ gtau_analytical atol=1e-9

        _, gtau_offdiag = calculate_green_tau(result, annh, beta, n_tau, 1, 2)
        @test all(isapprox.(gtau_offdiag, 0.0, atol=1e-9))
    end

    run_greens_test("Hubbard Atom",
        hopping_str=hopping_str,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn
    )
end

@testset "Green's Functions: Hubbard Dimer" begin
    t = 1.0
    U = 2.0
    beta = 20.0
    n_matsubara = 100
    
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
    
    verify_fn = (result, annh) -> begin
        beta = 20.0
        n_matsubara = 100
        
        iwn, g11_numeric = calculate_green_matsubara(result, annh, beta, n_matsubara, 1, 1)
        _, g12_numeric = calculate_green_matsubara(result, annh, beta, n_matsubara, 1, 3)

        c = sqrt(16*t^2 + U^2)
        c_minus_U = c - U
        a_sq = (32*t^2 + 2*c_minus_U^2) / c_minus_U^2
        
        term1_factor = (1 + 4*t / c_minus_U)^2
        term2_factor = (1 - 4*t / c_minus_U)^2

        g11_analytical = (1 / (2 * a_sq)) .* (
            (term1_factor ./ (iwn .+ t .- c/2)) .+
            (term2_factor ./ (iwn .- t .- c/2)) .+
            (term1_factor ./ (iwn .- t .+ c/2)) .+
            (term2_factor ./ (iwn .+ t .+ c/2))
        )

        g12_analytical = (1 / (2 * a_sq)) .* (
            (-term1_factor ./ (iwn .+ t .- c/2)) .+
            (term2_factor ./ (iwn .- t .- c/2)) .+
            (term1_factor ./ (iwn .- t .+ c/2)) .+
            (-term2_factor ./ (iwn .+ t .+ c/2))
        )
        
        @test g11_numeric ≈ g11_analytical atol=1e-6
        @test g12_numeric ≈ g12_analytical atol=1e-6
    end
    
    run_greens_test("Hubbard Dimer (Non-degenerate)",
        hopping_str=hopping_str,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn
    )
    
    # Almost Degenerate case
    t_deg = 0.001
    hopping_str_deg = """
    0 0 $(-U/2) 0.0
    1 1 $(-U/2) 0.0
    2 2 $(-U/2) 0.0
    3 3 $(-U/2) 0.0
    0 2 $(-t_deg) 0.0
    2 0 $(-t_deg) 0.0
    1 3 $(-t_deg) 0.0
    3 1 $(-t_deg) 0.0
    """
    
    verify_fn_deg = (result, annh) -> begin
        iwn, g11_numeric = calculate_green_matsubara(result, annh, beta, n_matsubara, 1, 1)
        _, g12_numeric = calculate_green_matsubara(result, annh, beta, n_matsubara, 1, 3)

        c = sqrt(16*t_deg^2 + U^2)
        c_minus_U = c - U
        a_sq = (32*t_deg^2 + 2*c_minus_U^2) / c_minus_U^2
        
        term1_factor = (1 + 4*t_deg / c_minus_U)^2
        term2_factor = (1 - 4*t_deg / c_minus_U)^2

        g11_analytical = (1 / (8 * a_sq)) .* (
            (term1_factor ./ (iwn .+ t_deg .- c/2)) .+
            (term2_factor ./ (iwn .- t_deg .- U/2)) .+
            (term1_factor ./ (iwn .- t_deg .+ U/2)) .+
            (term2_factor ./ (iwn .+ t_deg .+ U/2))
        ) .+ (3/16) .* (
            (1 ./ (iwn .+ t_deg .- U/2)) .+
            (1 ./ (iwn .- t_deg .- U/2)) .+
            (1 ./ (iwn .- t_deg .+ U/2)) .+
            (1 ./ (iwn .+ t_deg .+ U/2))
        )

        g12_analytical = (1 / (8 * a_sq)) .* (
            (-term1_factor ./ (iwn .+ t_deg .- c/2)) .+
            (term2_factor ./ (iwn .- t_deg .- U/2)) .+
            (term1_factor ./ (iwn .- t_deg .+ U/2)) .+
            (-term2_factor ./ (iwn .+ t_deg .+ U/2))
        ) .+ (3/16) .* (
            (1 ./ (iwn .+ t_deg .- U/2)) .+
            (-1 ./ (iwn .- t_deg .- U/2)) .+
            (-1 ./ (iwn .- t_deg .+ U/2)) .+
            (1 ./ (iwn .+ t_deg .+ U/2))
        )
        
        @test g11_numeric ≈ g11_analytical atol=1e-6
        @test g12_numeric ≈ g12_analytical atol=1e-6
    end
    
    run_greens_test("Hubbard Dimer (Almost Degenerate)",
        hopping_str=hopping_str_deg,
        coulomb_str=coulomb_str,
        site_qns_n=site_qns_n,
        site_qns_sz=site_qns_sz,
        verify_fn=verify_fn_deg
    )
end
