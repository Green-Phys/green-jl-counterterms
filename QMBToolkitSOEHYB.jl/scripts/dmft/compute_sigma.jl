using ArgParse
using Printf
using HDF5
using LinearAlgebra
using Lehmann

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--iteration"
            help = "Current DMFT iteration number"
            arg_type = Int
            required = true
        "--sim_h5"
            help = "Path to solver output sim.h5"
            arg_type = String
            required = true
        "--dmft_h5"
            help = "Path to DMFT data HDF5 file"
            arg_type = String
            required = true
        "--beta"
            help = "Inverse temperature"
            arg_type = Float64
            required = true
        "--dlr_euv"
            help = "DLR energy cutoff Euv"
            arg_type = Float64
            required = true
        "--dlr_rtol"
            help = "DLR relative tolerance"
            arg_type = Float64
            default = 1e-14
        "--n_matsubara"
            help = "Number of Matsubara frequencies for linear grid"
            arg_type = Int
            default = 1000
    end
    return parse_args(s)
end

# All arrays use solver convention: (n_flav, n_flav, n_points)

function compute_sigma(G0_iw::Array{ComplexF64, 3}, G_iw::Array{ComplexF64, 3})
    nf1, nf2, n_freq = size(G_iw)
    Sigma_iw = zeros(ComplexF64, nf1, nf2, n_freq)
    for k in 1:n_freq
        Sigma_iw[:, :, k] = inv(G0_iw[:, :, k]) - inv(G_iw[:, :, k])
    end
    return Sigma_iw
end

function main()
    args = parse_commandline()

    iteration = args["iteration"]
    sim_h5_path = args["sim_h5"]
    dmft_h5_path = args["dmft_h5"]
    beta = args["beta"]
    dlr_euv = args["dlr_euv"]
    dlr_rtol = args["dlr_rtol"]
    n_matsubara = args["n_matsubara"]

    println("=== compute_sigma.jl ===")
    @printf("Iteration: %d\n", iteration)
    @printf("n_matsubara (linear grid): %d\n", n_matsubara)

    # Create DLR grid
    dlrGrid = DLRGrid(dlr_euv, beta, dlr_rtol, true)
    @printf("DLR grid size: %d\n", dlrGrid.size)

    # Linear Matsubara grid: n = 0, 1, ..., n_matsubara-1
    linear_n = collect(0:n_matsubara-1)
    linear_freq = (2 .* linear_n .+ 1) .* π / beta

    # --- Read G0(iw) on DLR Matsubara grid from dmft.h5 ---
    # Shape: (n_flav, n_flav, n_freq_dlr)
    G0_iw_dlr, freq_G0 = h5open(dmft_h5_path, "r") do f
        group = "iter$(iteration)/dlr_grid"
        read(f, "$(group)/G0_iw"), read(f, "$(group)/freq")
    end

    if length(freq_G0) != length(dlrGrid.ωn) || !isapprox(freq_G0, dlrGrid.ωn; atol=1e-10)
        error("G0 Matsubara grid from dmft.h5 does not match DLR parameters")
    end

    # --- Read G(tau) on DLR tau grid from sim.h5, convert to Matsubara ---
    # sim.h5 stores (n_flav, n_flav, n_dlr) with grid axis last
    G_dlr_coeffs = h5open(sim_h5_path, "r") do f
        green_tau_dlr = read(f, "green/dlr_grid/green_tau")
        tau_dlr = read(f, "green/dlr_grid/tau")
        if length(tau_dlr) != length(dlrGrid.τ) || !isapprox(tau_dlr, dlrGrid.τ; atol=1e-10)
            error("DLR tau grid in sim.h5 does not match specified DLR parameters")
        end
        tau2dlr(dlrGrid, green_tau_dlr; axis=3, verbose=false)
    end

    G_iw_dlr = dlr2matfreq(dlrGrid, G_dlr_coeffs; axis=3, verbose=false)

    if size(G0_iw_dlr) != size(G_iw_dlr)
        error("Dimension mismatch: G0 $(size(G0_iw_dlr)) vs G $(size(G_iw_dlr))")
    end

    nf1, nf2, n_freq_dlr = size(G_iw_dlr)
    @printf("n_flavors: %d, DLR points: %d, linear points: %d\n", nf1, n_freq_dlr, n_matsubara)

    # --- Compute Sigma on DLR grid ---
    Sigma_iw_dlr = compute_sigma(G0_iw_dlr, G_iw_dlr)
    @printf("Sigma max abs (DLR): %.6e\n", maximum(abs.(Sigma_iw_dlr)))

    # --- Interpolate G0, G to linear Matsubara grid via DLR, then compute Sigma there ---
    # Note: Sigma has a constant (Hartree) term that DLR can't represent,
    # so we compute Sigma directly on the linear grid from G0 and G.
    G0_dlr_coeffs = matfreq2dlr(dlrGrid, G0_iw_dlr; axis=3, verbose=false)

    G0_iw_linear = dlr2matfreq(dlrGrid, G0_dlr_coeffs, linear_n; axis=3, verbose=false)
    G_iw_linear = dlr2matfreq(dlrGrid, G_dlr_coeffs, linear_n; axis=3, verbose=false)
    Sigma_iw_linear = compute_sigma(G0_iw_linear, G_iw_linear)

    # --- Write to dmft.h5 ---
    h5open(dmft_h5_path, "r+") do f
        group = "iter$(iteration)"

        # DLR grid subgroup — G0_iw and freq already written by write_hyb_G0.jl
        dlr_grp = f["$(group)/dlr_grid"]
        dlr_grp["G_iw"] = G_iw_dlr
        dlr_grp["Sigma_iw"] = Sigma_iw_dlr

        # Linear grid subgroup
        lin_grp = create_group(f, "$(group)/linear_grid")
        lin_grp["G0_iw"] = G0_iw_linear
        lin_grp["G_iw"] = G_iw_linear
        lin_grp["Sigma_iw"] = Sigma_iw_linear
        lin_grp["freq"] = linear_freq

        # Mark iteration as completed
        if haskey(f, "completed_iter")
            delete_object(f, "completed_iter")
        end
        f["completed_iter"] = iteration
    end

    println("Written to $(dmft_h5_path) under iter$(iteration)/:")
    println("  dlr_grid/: G0_iw, G_iw, Sigma_iw, freq")
    println("  linear_grid/: G0_iw, G_iw, Sigma_iw, freq")
    println("  completed_iter = $(iteration)")
    println("Done.")
end

main()
