#!/usr/bin/env julia
"""
Interpolate Green's function from DLR tau grid to Matsubara frequencies.

Reads Green's function on DLR tau grid from SOEHYB sim.h5 output,
reads Matsubara frequency grid from ED input h5 file,
and saves the interpolated Green's function on Matsubara frequencies.
"""

using ArgParse
using HDF5
using Lehmann

function freq_to_matsubara_index(freq, β)
    return @. round(Int, (freq * β / π - 1) / 2)
end

function parse_commandline()
    s = ArgParseSettings(description="Interpolate Green's function from DLR tau grid to Matsubara frequencies.")

    @add_arg_table! s begin
        "--sim_h5_file"
            help = "path to SOEHYB output sim.h5 file (contains Green's function on DLR tau grid)"
            required = true
        "--ed_h5_file"
            help = "path to ED input h5 file (contains Matsubara frequency grid)"
            required = true
        "--Euv"
            help = "DLR energy cutoff parameter"
            arg_type = Float64
            required = true
        "--beta"
            help = "inverse temperature"
            arg_type = Float64
            required = true
        "--dlr_tol"
            help = "DLR tolerance"
            arg_type = Float64
            default = 1e-14
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    sim_h5_file = args["sim_h5_file"]
    ed_h5_file = args["ed_h5_file"]
    Euv = args["Euv"]
    beta = args["beta"]
    dlr_tol = args["dlr_tol"]

    # Create DLR grid (fermionic)
    dlr_grid = DLRGrid(Euv, beta, dlr_tol, true)

    # Read Green's function on DLR tau grid from sim.h5
    # Julia HDF5 reads as (n_flavor, n_flavor, n_tau); permute to (n_tau, n_flavor, n_flavor)
    # for Lehmann.jl which operates along the first axis
    tau_grid, green_tau = h5open(sim_h5_file, "r") do f
        tau = read(f, "green/dlr_grid/tau")
        green = permutedims(read(f, "green/dlr_grid/green_tau"), (3, 2, 1))
        tau, green
    end

    @assert tau_grid ≈ dlr_grid.τ "DLR tau grid mismatch: file tau grid does not match DLR grid (check Euv, beta, dlr_tol)"

    # Read Matsubara frequency grid from ED file
    freq = h5open(ed_h5_file, "r") do f
        read(f, "freq")
    end

    # Convert frequencies to Matsubara indices: ωₙ = (2n+1)π/β → n = (ωₙβ/π - 1)/2
    matsubara_indices = freq_to_matsubara_index(freq, beta)

    # Interpolate: tau → DLR coefficients → Matsubara frequencies
    dlr_coeffs = tau2dlr(dlr_grid, green_tau, tau_grid)
    green_matsubara = dlr2matfreq(dlr_grid, dlr_coeffs, matsubara_indices)

    println("Interpolated Green's function: $(size(green_tau)) (tau) → $(size(green_matsubara)) (Matsubara)")

    # Save green_matsubara and freq to sim.h5
    # Permute back to (n_flavor, n_flavor, n_freq) so Python sees (n_freq, n_flavor, n_flavor)
    h5open(sim_h5_file, "r+") do f
        if haskey(f, "green/matsubara")
            delete_object(f, "green/matsubara")
        end
        matsubara_group = create_group(f, "green/matsubara")
        matsubara_group["green_iw"] = permutedims(green_matsubara, (3, 2, 1))
        matsubara_group["freq"] = freq
    end

    println("Saved to $sim_h5_file under green/matsubara/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
