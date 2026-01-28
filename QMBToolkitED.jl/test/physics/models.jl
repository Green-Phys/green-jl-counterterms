"""
Physics model definitions for validation testing.

Each model returns a NamedTuple with:
- name: Display name for test output
- hopping_str, coulomb_str: Hamiltonian input strings
- flavor_qns_n, flavor_qns_sz: Quantum numbers for QN_N and QN_NSz
- verify_Z: Optional (evals) -> tests on Hilbert space dimension and partition function
- verify_greens: Optional (result, annh) -> tests on Green's functions
"""

function physics_models()
    [
        chain_1d_model(),
        anderson_impurity_model(),
        hubbard_atom_model(),
        hubbard_dimer_model(),
        hubbard_dimer_degenerate_model(),
    ]
end

# --- Model 1: Non-interacting 1D Chain ---

function chain_1d_model()
    L = 3
    t = 2.0

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

    nflavors = 2 * L
    flavor_qns_n = [QN_N(1) for _ in 1:nflavors]
    flavor_qns_sz = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:nflavors]

    # Analytical single-particle spectrum
    H_hop_spin = [0.0 -t 0.0; -t 0.0 -t; 0.0 -t 0.0]
    E_spin = eigvals(H_hop_spin)

    H_hop = zeros(Float64, nflavors, nflavors)
    H_hop[1:2:end, 1:2:end] .= H_hop_spin
    H_hop[2:2:end, 2:2:end] .= H_hop_spin
    E_full, U_full = eigen(H_hop)

    verify_Z = (evals) -> begin
        beta = 10.0
        @test length(evals) == 2^nflavors

        Z_calculated = get_Z(evals, beta, shift_energy=true)
        E_ground = 2 * E_spin[1]
        Z_analytical = exp(beta * E_ground) * prod(1 .+ exp.(-beta .* E_spin))^2
        @test Z_calculated ≈ Z_analytical atol=1e-9
    end

    verify_greens = (result, annh) -> begin
        beta = 10.0
        n_matsubara = 100
        n_tau = 100

        # Matsubara
        iwn, g00 = calculate_green_matsubara(result, annh[1], annh[1], beta, n_matsubara)
        _, g02 = calculate_green_matsubara(result, annh[1], annh[3], beta, n_matsubara)
        _, g01 = calculate_green_matsubara(result, annh[1], annh[2], beta, n_matsubara)

        G_mat = zeros(ComplexF64, nflavors, nflavors, n_matsubara)
        for (idx, iw) in enumerate(iwn)
            G_mat[:, :, idx] = U_full * diagm([1.0 / (iw - E_full[k]) for k in 1:nflavors]) * U_full'
        end
        @test g00 ≈ G_mat[1, 1, :] atol=1e-9
        @test g02 ≈ G_mat[1, 3, :] atol=1e-9
        @test all(isapprox.(g01, 0.0, atol=1e-9))  # Spin conservation

        # Imaginary-time
        tau_grid, g00_tau = calculate_green_tau(result, annh[1], annh[1], beta, n_tau)
        _, g02_tau = calculate_green_tau(result, annh[1], annh[3], beta, n_tau)
        _, g01_tau = calculate_green_tau(result, annh[1], annh[2], beta, n_tau)

        G_tau = zeros(Float64, nflavors, nflavors, n_tau)
        for (idx, tau) in enumerate(tau_grid)
            g_k = [-begin
                if E_full[k] > 0
                    exp(-E_full[k] * tau) / (1 + exp(-beta * E_full[k]))
                else
                    exp(E_full[k] * (beta - tau)) / (1 + exp(beta * E_full[k]))
                end
            end for k in 1:nflavors]
            G_tau[:, :, idx] = real.(U_full * diagm(g_k) * U_full')
        end
        @test g00_tau ≈ G_tau[1, 1, :] atol=1e-9
        @test g02_tau ≈ G_tau[1, 3, :] atol=1e-9
        @test all(isapprox.(g01_tau, 0.0, atol=1e-9))  # Spin conservation
    end

    return (
        name = "1D Chain",
        hopping_str = hopping_str,
        coulomb_str = coulomb_str,
        flavor_qns_n = flavor_qns_n,
        flavor_qns_sz = flavor_qns_sz,
        verify_Z = verify_Z,
        verify_greens = verify_greens,
    )
end

# --- Model 2: Anderson Impurity ---

function anderson_impurity_model()
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

    flavor_qns_n = [QN_N(1) for _ in 1:6]
    flavor_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_Z = (evals) -> begin
        beta = 10.0
        @test length(evals) == 64
        Z = get_Z(evals, beta, shift_energy=true)
        @test Z ≈ 3.87936702286 atol=1e-9
    end

    return (
        name = "Anderson Impurity",
        hopping_str = hopping_str,
        coulomb_str = coulomb_str,
        flavor_qns_n = flavor_qns_n,
        flavor_qns_sz = flavor_qns_sz,
        verify_Z = verify_Z,
        verify_greens = nothing,
    )
end

# --- Model 3: Hubbard Atom ---

function hubbard_atom_model()
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

    flavor_qns_n = [QN_N(1), QN_N(1)]
    flavor_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_Z = (evals) -> begin
        beta = 10.0
        @test length(evals) == 4
        Z = get_Z(evals, beta, shift_energy=true)
        @test Z ≈ 2 * (1 + exp(-beta * U / 2)) atol=1e-9
    end

    verify_greens = (result, annh) -> begin
        beta = 10.0
        n_matsubara = 100
        n_tau = 100

        # Matsubara
        iwn, giwn = calculate_green_matsubara(result, annh[1], annh[1], beta, n_matsubara)
        @test giwn ≈ 0.5 .* (1 ./ (iwn .+ U / 2) .+ 1 ./ (iwn .- U / 2)) atol=1e-9

        _, giwn_off = calculate_green_matsubara(result, annh[1], annh[2], beta, n_matsubara)
        @test all(isapprox.(giwn_off, 0.0, atol=1e-9))

        # Imaginary-time
        tau_grid, gtau = calculate_green_tau(result, annh[1], annh[1], beta, n_tau)
        Z_an = 2 * (1 + exp(-beta * U / 2))
        @test gtau ≈ -(exp.(-(beta .- tau_grid) .* (U / 2)) .+ exp.(-tau_grid .* (U / 2))) ./ Z_an atol=1e-9

        _, gtau_off = calculate_green_tau(result, annh[1], annh[2], beta, n_tau)
        @test all(isapprox.(gtau_off, 0.0, atol=1e-9))
    end

    return (
        name = "Hubbard Atom",
        hopping_str = hopping_str,
        coulomb_str = coulomb_str,
        flavor_qns_n = flavor_qns_n,
        flavor_qns_sz = flavor_qns_sz,
        verify_Z = verify_Z,
        verify_greens = verify_greens,
    )
end

# --- Model 4: Hubbard Dimer ---

function hubbard_dimer_model()
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

    flavor_qns_n = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
    flavor_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_Z = (evals) -> begin
        beta = 10.0
        @test length(evals) == 16
        analytical_evals = [
            0.0, 0.0, 0.0,
            -U, -U, -U,
            -U/2 - t, -U/2 - t, -U/2 - t, -U/2 - t,
            -U/2 + t, -U/2 + t, -U/2 + t, -U/2 + t,
            0.5 * (-U - sqrt(16*t^2 + U^2)),
            0.5 * (-U + sqrt(16*t^2 + U^2))
        ]
        @test get_Z(evals, beta, shift_energy=true) ≈ get_Z(analytical_evals, beta, shift_energy=true) atol=1e-9
    end

    verify_greens = (result, annh) -> begin
        beta = 50.0
        n_matsubara = 100

        iwn, g11 = calculate_green_matsubara(result, annh[1], annh[1], beta, n_matsubara)
        _, g12 = calculate_green_matsubara(result, annh[1], annh[3], beta, n_matsubara)

        c = sqrt(16 * t^2 + U^2)
        c_minus_U = c - U
        a_sq = (32 * t^2 + 2 * c_minus_U^2) / c_minus_U^2
        term1 = (1 + 4 * t / c_minus_U)^2
        term2 = (1 - 4 * t / c_minus_U)^2

        g11_an = (1 / (2 * a_sq)) .* (
            term1 ./ (iwn .+ t .- c / 2) .+
            term2 ./ (iwn .- t .- c / 2) .+
            term1 ./ (iwn .- t .+ c / 2) .+
            term2 ./ (iwn .+ t .+ c / 2)
        )
        g12_an = (1 / (2 * a_sq)) .* (
            -term1 ./ (iwn .+ t .- c / 2) .+
            term2 ./ (iwn .- t .- c / 2) .+
            term1 ./ (iwn .- t .+ c / 2) .+
            -term2 ./ (iwn .+ t .+ c / 2)
        )

        @test g11 ≈ g11_an atol=1e-9
        @test g12 ≈ g12_an atol=1e-9
    end

    return (
        name = "Hubbard Dimer",
        hopping_str = hopping_str,
        coulomb_str = coulomb_str,
        flavor_qns_n = flavor_qns_n,
        flavor_qns_sz = flavor_qns_sz,
        verify_Z = verify_Z,
        verify_greens = verify_greens,
    )
end

# --- Model 5: Hubbard Dimer (Near-Degenerate) ---

function hubbard_dimer_degenerate_model()
    t = 0.001
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

    flavor_qns_n = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
    flavor_qns_sz = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

    verify_Z = (evals) -> begin
        beta = 10.0
        @test length(evals) == 16
        analytical_evals = [
            0.0, 0.0, 0.0,
            -U, -U, -U,
            -U/2 - t, -U/2 - t, -U/2 - t, -U/2 - t,
            -U/2 + t, -U/2 + t, -U/2 + t, -U/2 + t,
            0.5 * (-U - sqrt(16*t^2 + U^2)),
            0.5 * (-U + sqrt(16*t^2 + U^2))
        ]
        @test get_Z(evals, beta, shift_energy=true) ≈ get_Z(analytical_evals, beta, shift_energy=true) atol=1e-9
    end

    verify_greens = (result, annh) -> begin
        beta = 50.0
        n_matsubara = 100

        iwn, g11 = calculate_green_matsubara(result, annh[1], annh[1], beta, n_matsubara)
        _, g12 = calculate_green_matsubara(result, annh[1], annh[3], beta, n_matsubara)

        c = sqrt(16 * t^2 + U^2)
        c_minus_U = c - U
        a_sq = (32 * t^2 + 2 * c_minus_U^2) / c_minus_U^2
        term1 = (1 + 4 * t / c_minus_U)^2
        term2 = (1 - 4 * t / c_minus_U)^2

        exp_base = exp(beta * (U / 2 - c / 2))
        exp_pos_t = exp(beta * (t - c / 2))
        exp_neg_t = exp(beta * (-t - c / 2))
        denom = 1 + 3 * exp_base

        f1 = (1 + exp_pos_t) / denom
        f2 = (1 + exp_neg_t) / denom
        f3 = (exp_base + exp_pos_t) / denom
        f4 = (exp_base + exp_neg_t) / denom

        g11_an = (f1 / (2 * a_sq)) .* (
            term1 ./ (iwn .+ t .- c / 2) .+
            term1 ./ (iwn .- t .+ c / 2)
        ) .+ (f2 / (2 * a_sq)) .* (
            term2 ./ (iwn .- t .- c / 2) .+
            term2 ./ (iwn .+ t .+ c / 2)
        ) .+ (3 * f3 / 4) .* (
            1 ./ (iwn .+ t .- U / 2) .+
            1 ./ (iwn .- t .+ U / 2)
        ) .+ (3 * f4 / 4) .* (
            1 ./ (iwn .- t .- U / 2) .+
            1 ./ (iwn .+ t .+ U / 2)
        )

        g12_an = (f1 / (2 * a_sq)) .* (
            -term1 ./ (iwn .+ t .- c / 2) .+
            term1 ./ (iwn .- t .+ c / 2)
        ) .+ (f2 / (2 * a_sq)) .* (
            term2 ./ (iwn .- t .- c / 2) .+
            -term2 ./ (iwn .+ t .+ c / 2)
        ) .+ (3 * f3 / 4) .* (
            1 ./ (iwn .+ t .- U / 2) .+
            -1 ./ (iwn .- t .+ U / 2)
        ) .+ (3 * f4 / 4) .* (
            -1 ./ (iwn .- t .- U / 2) .+
            1 ./ (iwn .+ t .+ U / 2)
        )

        @test g11 ≈ g11_an atol=1e-9
        @test g12 ≈ g12_an atol=1e-9
    end

    return (
        name = "Hubbard Dimer (Near-Degenerate)",
        hopping_str = hopping_str,
        coulomb_str = coulomb_str,
        flavor_qns_n = flavor_qns_n,
        flavor_qns_sz = flavor_qns_sz,
        verify_Z = verify_Z,
        verify_greens = verify_greens,
    )
end
