using ArgParse
using Printf
using HDF5
using Lehmann
using LinearAlgebra
using DelimitedFiles

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--beta"
            help = "Inverse temperature"
            arg_type = Float64
            required = true
        "--t"
            help = "Bethe lattice hopping parameter (delta = t^2 * G)"
            arg_type = Float64
            default = 1.0
        "--n_spin"
            help = "Number of spin flavors"
            arg_type = Int
            required = true
        "--n_orbital"
            help = "Number of orbitals"
            arg_type = Int
            required = true
        "--n_tau"
            help = "Number of tau grid points for output"
            arg_type = Int
            required = true
        "--iteration"
            help = "Current DMFT iteration number"
            arg_type = Int
            required = true
        "--prev_sim_h5"
            help = "Path to previous iteration's sim.h5 (required if iteration > 1)"
            arg_type = String
            default = ""
        "--output_file"
            help = "Output path for delta.txt"
            arg_type = String
            required = true
        "--off_diagonal_prefactor"
            help = "Prefactor for off-diagonal elements of semicircular hybridization (iteration 1 only)"
            arg_type = Float64
            default = 0.0
        "--zero_off_diagonal"
            help = "If true, set all off-diagonal hybridization to zero before writing"
            arg_type = Bool
            default = false
        "--hopping_file"
            help = "Path to hopping matrix file (includes mu, no counterterms)"
            arg_type = String
            required = true
        "--dlr_euv"
            help = "DLR energy cutoff Euv"
            arg_type = Float64
            required = true
        "--dlr_rtol"
            help = "DLR relative tolerance"
            arg_type = Float64
            default = 1e-14
        "--dmft_h5"
            help = "Output HDF5 file for DMFT data (G0_iw stored under iter{x}/ group)"
            arg_type = String
            required = true
    end
    return parse_args(s)
end

function gen_semicircular_delta(n_tau::Int, beta::Float64, n_spin::Int, n_orbital::Int, t::Float64; off_diagonal_prefactor::Float64=0.0)
    tau_grid = collect(LinRange(0.0, beta, n_tau))
    dlrGrid = DLRGrid(2.0, beta, 1e-14, true)
    G_tau = -1 * 2 / pi * Sample.SemiCircle(dlrGrid, :τ, tau_grid)

    n_flavors = n_spin * n_orbital
    delta_tau = zeros(Float64, n_tau, n_flavors, n_flavors)

    tsq = t^2
    for o1 in 1:n_orbital
        for s1 in 1:n_spin
            i = (o1 - 1) * n_spin + s1
            for o2 in 1:n_orbital
                for s2 in 1:n_spin
                    j = (o2 - 1) * n_spin + s2
                    if i == j
                        delta_tau[:, i, j] = tsq .* G_tau
                    elseif s1 == s2
                        delta_tau[:, i, j] = tsq * off_diagonal_prefactor .* G_tau
                    end
                end
            end
        end
    end

    return tau_grid, delta_tau
end

function read_green_from_sim_h5(sim_h5_path::String, n_tau::Int, beta::Float64, n_spin::Int, n_orbital::Int)
    n_flavors = n_spin * n_orbital

    h5open(sim_h5_path, "r") do f
        tau = read(f, "green/linear_grid/tau")
        green_tau = read(f, "green/linear_grid/green_tau")
        beta_file = read(f, "model/beta")

        # Check beta consistency
        if !isapprox(beta_file, beta; atol=1e-10)
            error("Beta mismatch: sim.h5 has beta=$(beta_file) but --beta=$(beta)")
        end

        # Check tau grid size consistency
        n_tau_file = length(tau)
        if n_tau_file != n_tau
            error("Tau grid size mismatch: sim.h5 has $(n_tau_file) points but --n_tau=$(n_tau)")
        end

        # Check tau grid values consistency
        tau_expected = collect(LinRange(0.0, beta, n_tau))
        if !isapprox(tau, tau_expected; atol=1e-10)
            error("Tau grid values from sim.h5 do not match expected LinRange(0, $(beta), $(n_tau))")
        end

        # Check flavor dimension consistency
        n_tau_g, nf1, nf2 = size(green_tau)
        if nf1 != n_flavors || nf2 != n_flavors
            error("Flavor dimension mismatch: sim.h5 Green's function has shape ($n_tau_g, $nf1, $nf2) " *
                  "but expected ($n_tau, $n_flavors, $n_flavors) from n_spin=$n_spin * n_orbital=$n_orbital = $n_flavors")
        end

        return tau, green_tau
    end
end

function write_delta_txt(output_file::String, tau_grid::Vector{Float64}, delta_tau::Array{<:Number, 3})
    n_tau, n_flav1, n_flav2 = size(delta_tau)
    open(output_file, "w") do io
        for t in 1:n_tau
            for i in 1:n_flav1
                for j in 1:n_flav2
                    val = delta_tau[t, i, j]
                    @printf(io, "%d %d %d %.18e %.18e\n",
                        t - 1, i - 1, j - 1, real(val), imag(val))
                end
            end
        end
    end
end

function zero_off_diagonal!(delta_tau::Array{<:Number, 3})
    n_tau, nf1, nf2 = size(delta_tau)
    for t in 1:n_tau
        for i in 1:nf1
            for j in 1:nf2
                if i != j
                    delta_tau[t, i, j] = zero(eltype(delta_tau))
                end
            end
        end
    end
end

function read_hopping_matrix(hopping_file::String, n_flavors::Int)
    data = readdlm(hopping_file)
    @assert size(data, 2) == 4 "Hopping file must have 4 columns: ind_orb_1, ind_orb_2, Re, Im"
    n_orb = Int(maximum(data[:, 1:2])) + 1
    if n_orb != n_flavors
        error("Hopping matrix dimension ($n_orb) does not match n_flavors ($n_flavors) = n_spin * n_orbital")
    end
    @assert size(data, 1) == n_orb * n_orb "Hopping file rows ($(size(data, 1))) must equal n_flavors^2 ($(n_orb * n_orb))"
    complex_flat = complex.(data[:, 3], data[:, 4])
    H = permutedims(reshape(complex_flat, n_orb, n_orb))
    return H
end

function check_dlr_consistency(sim_h5_path::String, dlrGrid::DLRGrid)
    h5open(sim_h5_path, "r") do f
        tau_dlr_file = read(f, "green/dlr_grid/tau")
        tau_dlr_expected = dlrGrid.τ
        if length(tau_dlr_file) != length(tau_dlr_expected)
            error("DLR grid size mismatch: sim.h5 has $(length(tau_dlr_file)) DLR points " *
                  "but specified DLR parameters give $(length(tau_dlr_expected))")
        end
        if !isapprox(tau_dlr_file, tau_dlr_expected; atol=1e-10)
            error("DLR tau grid values from sim.h5 do not match those from specified DLR parameters (Euv=$(dlrGrid.Euv), rtol=$(dlrGrid.rtol))")
        end
    end
end

function compute_G0_iw(delta_tau::Array{<:Number, 3}, tau_grid::Vector{Float64}, H::Matrix{ComplexF64}, dlrGrid::DLRGrid; zero_off_diagonal::Bool=false)
    n_flavors = size(H, 1)
    beta = dlrGrid.β

    # delta(tau) -> DLR coefficients -> delta(iw) on DLR Matsubara grid
    dlr_coeffs = tau2dlr(dlrGrid, delta_tau, tau_grid; axis=1, verbose=false)
    delta_iw = dlr2matfreq(dlrGrid, dlr_coeffs; axis=1, verbose=false)

    if zero_off_diagonal
        for k in axes(delta_iw, 1)
            for i in 1:n_flavors
                for j in 1:n_flavors
                    if i != j
                        delta_iw[k, i, j] = zero(eltype(delta_iw))
                    end
                end
            end
        end
    end

    freq = dlrGrid.ωn
    n_freq = length(freq)
    G0_iw = zeros(ComplexF64, n_freq, n_flavors, n_flavors)
    I_mat = Matrix{ComplexF64}(I, n_flavors, n_flavors)

    for k in 1:n_freq
        G0_inv = 1im * freq[k] * I_mat - H - delta_iw[k, :, :]
        G0_iw[k, :, :] = inv(G0_inv)
    end

    return G0_iw, freq
end

function write_G0_to_h5(dmft_h5::String, iteration::Int, G0_iw::Array{ComplexF64, 3}, freq::Vector{Float64})
    h5open(dmft_h5, "cw") do f
        group_name = "iter$(iteration)"
        if haskey(f, group_name)
            delete_object(f, group_name)
        end
        g = create_group(f, group_name)
        g["G0_iw"] = G0_iw
        g["freq"] = freq
    end
end

function main()
    args = parse_commandline()

    beta = args["beta"]
    t = args["t"]
    n_spin = args["n_spin"]
    n_orbital = args["n_orbital"]
    n_flavors = n_spin * n_orbital
    n_tau = args["n_tau"]
    iteration = args["iteration"]
    prev_sim_h5 = args["prev_sim_h5"]
    output_file = args["output_file"]
    off_diagonal_prefactor = args["off_diagonal_prefactor"]
    do_zero_off_diag = args["zero_off_diagonal"]
    hopping_file = args["hopping_file"]
    dlr_euv = args["dlr_euv"]
    dlr_rtol = args["dlr_rtol"]
    dmft_h5 = args["dmft_h5"]

    println("=== write_hyb_G0.jl ===")
    @printf("Iteration: %d\n", iteration)
    @printf("Beta: %.6f, t: %.6f, t^2: %.6f\n", beta, t, t^2)
    @printf("n_spin: %d, n_orbital: %d, n_flavors: %d\n", n_spin, n_orbital, n_flavors)
    @printf("n_tau: %d\n", n_tau)
    @printf("DLR Euv: %.6f, rtol: %.2e\n", dlr_euv, dlr_rtol)
    @printf("Output delta: %s\n", output_file)
    @printf("Output DMFT H5: %s\n", dmft_h5)

    # Read hopping matrix and check dimensions
    @printf("Reading hopping from: %s\n", hopping_file)
    H = read_hopping_matrix(hopping_file, n_flavors)

    # Create DLR grid
    dlrGrid = DLRGrid(dlr_euv, beta, dlr_rtol, true)
    @printf("DLR grid size: %d\n", dlrGrid.size)

    # For iteration > 1, check DLR consistency with sim.h5
    if iteration > 1
        if prev_sim_h5 == ""
            error("--prev_sim_h5 is required for iteration > 1")
        end
        println("Checking DLR consistency with sim.h5...")
        check_dlr_consistency(prev_sim_h5, dlrGrid)
        println("DLR consistency check passed.")
    end

    # Generate delta(tau)
    if iteration == 1
        println("Using semicircular DOS for initial hybridization")
        @printf("Off-diagonal prefactor: %.6f\n", off_diagonal_prefactor)
        tau_grid, delta_tau = gen_semicircular_delta(n_tau, beta, n_spin, n_orbital, t;
            off_diagonal_prefactor=off_diagonal_prefactor)
    elseif iteration > 1
        @printf("Reading G(tau) from: %s\n", prev_sim_h5)
        tau_grid, green_tau = read_green_from_sim_h5(prev_sim_h5, n_tau, beta, n_spin, n_orbital)
        delta_tau = t^2 .* green_tau
    else
        error("Iteration must be >= 1, got $iteration")
    end

    if do_zero_off_diag
        println("Zeroing all off-diagonal elements")
        zero_off_diagonal!(delta_tau)
    end

    @printf("Delta max abs value: %.6e\n", maximum(abs.(delta_tau)))
    @printf("Delta max abs imaginary part: %.6e\n", maximum(abs.(imag.(delta_tau))))

    # Write delta.txt
    write_delta_txt(output_file, tau_grid, delta_tau)
    println("Delta written to $(output_file)")

    # Compute G0(iw) and write to DMFT H5
    println("Computing G0(iw) on DLR Matsubara grid...")
    G0_iw, freq = compute_G0_iw(delta_tau, tau_grid, H, dlrGrid; zero_off_diagonal=do_zero_off_diag)
    @printf("G0 Matsubara grid: %d frequencies\n", length(freq))
    write_G0_to_h5(dmft_h5, iteration, G0_iw, freq)
    println("G0(iw) written to $(dmft_h5) under iter$(iteration)/")

    println("Done.")
end

main()
