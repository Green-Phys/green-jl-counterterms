using ArgParse
using Printf
using LinearAlgebra
using DelimitedFiles
using HDF5

using PyCall
println("Python version info (to use adapol library):")
println("Version: ", PyCall.pyversion)
println("Executable: ", PyCall.pyprogramname)
println("Library: ", PyCall.libpython)
apapol_fitting = pyimport("adapol").pole_fitting

using NumericsSOE
using Lehmann # for SemiCircular hybridization function and DLR

function _write_hyb_weight(hyb_weight::Array{Float64, 3}, file_out::String)
    """
    Writes a 3D complex array to a flat file using efficient streaming.
    """
    n_poles, n_orb_1, n_orb_2 = size(hyb_weight)
    @assert n_orb_1 == n_orb_2 "Orbital dimensions must match"
    n_orb = n_orb_1

    # Open the file for writing
    open(file_out, "w") do io
        for i in 1:n_poles
            for j in 1:n_orb
                for k in 1:n_orb
                    val = hyb_weight[i, j, k]
                    @printf(io, "%d %d %d %.18e\n",
                        i-1, j-1, k-1, val)
                end
            end
        end
    end
end

function _write_hyb_poles(hyb_poles::Vector{Float64}, file_out::String)
    """
    Writes a 1D array of pole energies to a flat file.
    """
    open(file_out, "w") do io
        for i in 1:length(hyb_poles)
            @printf(io, "%.18e\n", hyb_poles[i])
        end
    end
end

"""
    read_hopping(hopping::Matrix{Float64}) -> Matrix{ComplexF64}

Reads a flat 2D array (N*N x 4) into a 2D complex array (N x N) using vectorized operations.
"""
function read_hopping(hopping::Matrix{Float64})
    # 1. Assert dimensions
    # In Julia, size(A, 2) gets the number of columns
    @assert size(hopping, 2) == 4 "Input must have 4 columns: ind_orb_1, ind_orb_2, Re, Im"

    # 2. Find dimensions
    # Python columns 0 and 1 correspond to Julia columns 1 and 2
    # We assume the file has 0-based indices, so we add 1 to find the count
    n_orb = Int(maximum(hopping[:, 1:2])) + 1
    @assert size(hopping, 1) == n_orb * n_orb "Rows must equal n_orb * n_orb"

    # --- Vectorized Part ---

    # 3. Create 1D complex array
    # Python col 2 (Re) is Julia col 3
    # Python col 3 (Im) is Julia col 4
    complex_flat = complex.(hopping[:, 3], hopping[:, 4])

    # 4. Reshape
    # Julia fills Column-Major by default. 
    # To match Python's Row-Major reshape, we reshape to (n_orb, n_orb) 
    # and then Permute Dimensions (Transpose) to align the data correctly.
    Hopping = permutedims(reshape(complex_flat, n_orb, n_orb))

    return Hopping
end

"""
    write_hopping(Hopping::Matrix{ComplexF64}, file_out::String)

Writes a 2D complex array to a flat file, matching the Python output format.
"""
function write_hopping(Hopping::Matrix{ComplexF64}, file_out::String)
    n_orb_1, n_orb_2 = size(Hopping)
    @assert n_orb_1 == n_orb_2 "Matrix must be square"
    n_orb = n_orb_1
    n_total = n_orb * n_orb

    # Create output array
    data_out = Matrix{Float64}(undef, n_total, 4)

    # --- Vectorized Part ---

    # 1. Fill data columns (Re and Im)
    # Python's .ravel() flattens Row-by-Row. 
    # Julia's vec() flattens Col-by-Col.
    # We must transpose (permutedims) first to flatten in the same order as Python.
    flat_hopping = vec(permutedims(Hopping))

    data_out[:, 3] = real.(flat_hopping) # Python col 2 is Julia col 3
    data_out[:, 4] = imag.(flat_hopping) # Python col 3 is Julia col 4

    # 2. Generate index columns (i, j)
    # We generate 0-based indices to match the Python output file format exactly.

    # i (orb_1, slowest changing index -> Row index)
    # repeat([0, 1], inner=2) -> [0, 0, 1, 1]
    data_out[:, 1] = repeat(0:(n_orb - 1), inner = n_orb)

    # j (orb_2, fastest changing index -> Col index)
    # repeat([0, 1], outer=2) -> [0, 1, 0, 1]
    data_out[:, 2] = repeat(0:(n_orb - 1), outer = n_orb)

    # 3. Save to file
    # We use open() with a do-block for safe file handling
    open(file_out, "w") do io
        for i in 1:size(data_out, 1)
            # Matching format: '%.0f', '%.0f', '%.18e', '%.18e'
            @printf(io, "%.0f %.0f %.18e %.18e\n",
                data_out[i, 1], data_out[i, 2], data_out[i, 3], data_out[i, 4])
        end
    end
end

function _obtain_hyb(poles, weights; svdtol = 1e-7)
    # From the output (poles, weights) to (counterterms_energy, counterterms_hyb)
    # poles shape: (p)
    # weights shape: (p, o1, o2)
    # counterterms_energy [E]: (Nb)
    # counterterms_hyb [V]: (Nb, o)

    # Delta_{ij}(tau) = \sum_{k} weights_{ijk} K(poles_{k}, tau) = \sum_{b} -1.0 * V_{bi}^{*} V_{bj} K(E_{b}, tau)
    counterterms_energy = []
    counterterms_hyb = []

    if length(poles) == 0
        return Float64[], Array{Float64, 2}(undef, 0, size(weights, 2))
    end

    # Loop over each matrix in the 'weights' list
    for i in 1:size(weights, 1)
        E = poles[i]
        sum_b_V_bi_V_bj = -1.0 * weights[i, :, :]

        eigvals, eigvecs = eigen(Hermitian(sum_b_V_bi_V_bj))

        # Loop over each eigenvalue/eigenvector pair
        for j in 1:length(eigvals)
            if eigvals[j] > svdtol
                push!(counterterms_energy, E)
                push!(counterterms_hyb, eigvecs[:, j] * sqrt(eigvals[j]))
            end
        end
    end
    # make veclist into a matrix with same number of rows as counterterms_energy
    counterterms_hyb = hcat(counterterms_hyb...)'

    return real.(counterterms_energy), real.(counterterms_hyb)
end

function _obtain_hyb_general(poles, weights; svdtol = 1e-7, separate_spins = false)
    # Wrapper for _obtain_hyb that handles separate_spins option
    # When separate_spins=true, process spin-up and spin-down separately and interleave results
    # Input format when separate_spins=true:
    #   poles: [up_1, dn_1, up_2, dn_2, ...] (alternating)
    #   weights: block-diagonal with spin-up at odd indices, spin-down at even indices
    # Output format when separate_spins=true:
    #   counterterms_energy: [up_1, dn_1, up_2, dn_2, ...] (alternating)
    #   counterterms_hyb: coupling vectors with spin structure preserved

    if !separate_spins
        return _obtain_hyb(poles, weights; svdtol = svdtol)
    end

    n_flavors = size(weights, 2)
    n_poles = length(poles)

    if n_poles == 0
        return Float64[], Array{Float64, 2}(undef, 0, n_flavors)
    end

    # Extract spin-up and spin-down poles/weights
    # Spin-up: odd indices (1, 3, 5, ...)
    # Spin-down: even indices (2, 4, 6, ...)
    up_indices = 1:2:n_poles
    dn_indices = 2:2:n_poles
    flavor_indices_up = collect(1:2:n_flavors)
    flavor_indices_dn = collect(2:2:n_flavors)

    poles_up = poles[up_indices]
    poles_dn = poles[dn_indices]
    # Extract the spin-up block (only the spin-up flavors)
    weights_up = weights[up_indices, flavor_indices_up, flavor_indices_up]
    weights_dn = weights[dn_indices, flavor_indices_dn, flavor_indices_dn]

    # Process each spin sector separately
    energy_up, hyb_up = _obtain_hyb(poles_up, weights_up; svdtol = svdtol)
    energy_dn, hyb_dn = _obtain_hyb(poles_dn, weights_dn; svdtol = svdtol)

    @assert length(energy_up) == length(energy_dn) "Spin up and spin down must have the same number of bath orbitals"

    n_bath = length(energy_up)
    if n_bath == 0
        return Float64[], Array{Float64, 2}(undef, 0, n_flavors)
    end

    # Interleave results: [up_1, dn_1, up_2, dn_2, ...]
    counterterms_energy = zeros(Float64, 2 * n_bath)
    counterterms_hyb = zeros(Float64, 2 * n_bath, n_flavors)

    for i in 1:n_bath
        # Spin-up bath orbital at odd index
        counterterms_energy[2 * i - 1] = energy_up[i]
        counterterms_hyb[2 * i - 1, flavor_indices_up] = hyb_up[i, :]

        # Spin-down bath orbital at even index
        counterterms_energy[2 * i] = energy_dn[i]
        counterterms_hyb[2 * i, flavor_indices_dn] = hyb_dn[i, :]
    end

    return counterterms_energy, counterterms_hyb
end

function _kernel_tau(tau, omega, beta)
    tau=Float64(tau)
    omega=Float64(omega)
    beta=Float64(beta)
    @assert tau >= -beta && tau <= beta "tau must be in [-beta, beta]"
    if tau >= 0.0
        if omega > 0.0
            return exp(-tau * omega)/(1 + exp(-beta * omega))
        else
            return exp(-(tau - beta) * omega)/(1 + exp(beta * omega))
        end
    else
        if omega > 0.0
            return -1 * exp(-(tau + beta) * omega)/(1 + exp(-beta * omega))
        else
            return -1 * exp(-tau * omega)/(1 + exp(beta * omega))
        end
    end
end

function _reconstruct_green_hyb(counterterms_energy::Vector{Float64}, counterterms_hyb::Matrix{Float64}, tau_grid::Vector{Float64}, beta::Float64)
    if length(counterterms_energy) == 0
        n_tau = length(tau_grid)
        n_flavors = size(counterterms_hyb, 2)
        return zeros(Float64, n_tau, n_flavors, n_flavors)
    end
    # Reconstruct hybridization function from bath parameters and compute inf-norm error with target hybridization function
    n_tau = length(tau_grid)
    n_flavors = size(counterterms_hyb, 2)
    reconstructed_hyb = zeros(Float64, n_tau, n_flavors, n_flavors)
    for (k, tau) in enumerate(tau_grid)
        for b in 1:length(counterterms_energy)
            K_val = _kernel_tau(tau, counterterms_energy[b], beta)
            reconstructed_hyb[k, :, :] .+= -1.0 * counterterms_hyb[b, :] * counterterms_hyb[b, :]' * K_val
        end
    end
    return reconstructed_hyb
end

function _reconstruct_green(poles::Vector{Float64}, weights::Array{Float64, 3}, tau_grid::Vector{Float64}, beta::Float64)
    # sum_{i} weights[i] * K(tau, poles[i]), different matrix element share the same poles
    n_tau = length(tau_grid)
    p = size(weights, 1)
    o1 = size(weights, 2)
    o2 = size(weights, 3)
    val = zeros(Float64, n_tau, o1, o2)
    for (k, tau) in enumerate(tau_grid)
        for j in 1:p
            K_val = _kernel_tau(tau, poles[j], beta)
            val[k, :, :] .+= weights[j, :, :] * K_val
        end
    end
    return val
end

function _inf_error(green::Array{Float64, 3}, reconstructed_green::Array{Float64, 3})
    return maximum(abs.(green .- reconstructed_green))
end

#! format: off
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--beta"
        help = "inverse temperature"
        arg_type = Float64
        required = true
        "--n_spin_impurity"
        help = "number of spin flavors for the impurity"
        arg_type = Int
        required = true
        "--n_orbital_impurity"
        help = "number of orbitals for the impurity"
        arg_type = Int
        required = true
        "--n_tau"
        help = "number of tau points for fitting"
        arg_type = Int
        required = true
        "--n_matsubara"
        help = "number of Matsubara frequencies for fitting"
        arg_type = Int
        required = true
        "--dlr_eps_counterterms"
        help = "DLR epsilon for counterterms fitting (only used if SOE subroutine is DLR)"
        arg_type = Float64
        default = 1e-14
        "--dlr_eps_hybridization"
        help = "DLR epsilon for hybridization fitting (only used if SOE subroutine is DLR)"
        arg_type = Float64
        default = 1e-14
        "--hyb_type"
        help = "type of hybridization function: 'SemiCircular' or 'DiscretePoles'"
        arg_type = String
        required = true
        "--hopping_file"
        help = "file name for the hopping matrix to be modified"
        arg_type = String
        required = true
        "--soe_subroutine_counterterms"
        help = "Type of SOE subroutine for counterterms: 'ESPRIT' or 'ADAPOL' or 'DLR'"
        arg_type = String
        required = true
        "--n_soe_counterterms"
        help = "Number of SOE terms for counterterms"
        arg_type = Int
        required = true
        "--soe_subroutine_hyb"
        help = "Type of SOE subroutine for hybridization: 'ESPRIT' or 'ADAPOL'"
        arg_type = String
        required = true
        "--n_soe_hybridization"
        help = "Number of SOE terms for hybridization"
        arg_type = Int
        required = true
        "--output_dir"
        help = "Output directory for generated files"
        arg_type = String
        required = true
        "--do_subtraction_trick"
        help = "Whether to do subtraction trick (For ESPRIT or ADAPOL, the remaining hybrdization is not well fitted, but one can fit the original data at different precision, the remaining hybridizaiotn function is a sum of them with the high precision part has negative coefficients)"
        arg_type = Bool
        default = false
        "--off_diagonal_prefactor"
        help = "Prefactor for off-diagonal elements of hybridization function"
        arg_type = Float64
        default = 0.0
        "--hyb_file"
        help = "Path to HDF5 file containing hybridization function (required for hyb_type='from_seet')"
        arg_type = String
        default = ""
        "--init_dlr_euv"
        help = "Initial DLR Euv parameter for reading hybridization from file (only used if hyb_type='from_seet')"
        arg_type = Float64
        default = 2.0
        "--separate_spins"
        help = "Whether to fit spin up and spin down separately (only supported for ESPRIT)"
        arg_type = Bool
        default = false
    end
    return parse_args(s)
end


function print_input(args)
    println("--- Input parameters ---")
    @printf("Beta: %.6f\n", args["beta"])
    @printf("Number of spin flavors for impurity: %d\n", args["n_spin_impurity"])
    @printf("Number of orbitals for impurity: %d\n", args["n_orbital_impurity"])
    @printf("Number of tau points: %d\n", args["n_tau"])
    @printf("Number of Matsubara frequencies: %d\n", args["n_matsubara"])
    @printf("Type of hybridization function: %s\n", args["hyb_type"])
    @printf("Hopping file to be modified: %s\n", args["hopping_file"])
    @printf("SOE subroutine for counterterms: %s\n", args["soe_subroutine_counterterms"])
    @printf("Number of SOE terms for counterterms: %d\n", args["n_soe_counterterms"])
    @printf("DLR epsilon for counterterms: %.6e\n", args["dlr_eps_counterterms"])
    @printf("SOE subroutine for hybridization: %s\n", args["soe_subroutine_hyb"])
    @printf("Number of SOE terms for hybridization: %d\n", args["n_soe_hybridization"])
    @printf("DLR epsilon for hybridization: %.6e\n", args["dlr_eps_hybridization"])
    @printf("Output directory: %s\n", args["output_dir"])
    @printf("Do subtraction trick: %s\n", args["do_subtraction_trick"])
    @printf("Separate spins: %s\n", args["separate_spins"])
    @printf("Prefactor for off-diagonal elements: %f\n", args["off_diagonal_prefactor"])
    if args["hyb_file"] != ""
        @printf("Hybridization file: %s\n", args["hyb_file"])
        @printf("Initial DLR Euv: %f\n", args["init_dlr_euv"])
    end
end

struct GData
    G_tau::Array{Float64, 3}
    tau_grid::Vector{Float64}
    G_iwn::Array{ComplexF64, 3}
    wn_grid::Vector{Float64}
end

function _from_spin_orbital_to_flavor(f_so)
    # Assuming f_so has shape (n, n_spin, n_orb, n_orb)
    n, n_spin, n_orb, _ = size(f_so)
    n_flavor = n_spin * n_orb

    f = zeros(ComplexF64, n, n_flavor, n_flavor)

    # spin runs faster
    for spin in 1:n_spin
        for orb1 in 1:n_orb
            for orb2 in 1:n_orb
                flavor1 = n_spin * (orb1 - 1) + spin
                flavor2 = n_spin * (orb2 - 1) + spin
                f[:, flavor1, flavor2] = f_so[:, spin, orb1, orb2]
            end
        end
    end

    return f
end

function _freq_to_matsubara_index(freq, β)
    return @. round(Int, (freq * β / π - 1) / 2)
end

function _read_delta_iw_from_h5(filename, beta)
    h5open(filename, "r") do f
        freq = read(f, "freq")
        # Reverse dimensions to match Python convention: (n_freq, n_spin, n_orb, n_orb)
        delta_iw_in = permutedims(read(f, "Delta/data_in"), (4, 3, 2, 1))
        delta_iw = _from_spin_orbital_to_flavor(delta_iw_in)
        matsubara_indices = _freq_to_matsubara_index(freq, beta)
        return matsubara_indices, delta_iw
    end
end

function _read_delta_tau_from_txt(filename, beta)
    data = readdlm(filename)

    # All indices in file are 0-based
    tau_indices = Int.(data[:, 1])
    i_indices = Int.(data[:, 2])
    j_indices = Int.(data[:, 3])
    real_parts = data[:, 4]

    n_tau = maximum(tau_indices) + 1
    n_flav = max(maximum(i_indices), maximum(j_indices)) + 1

    # Convert tau indices to actual tau values: [0, beta]
    tau_grid = collect(0:n_tau-1) .* (beta / (n_tau - 1))

    delta_tau = zeros(Float64, n_tau, n_flav, n_flav)

    for k in 1:size(data, 1)
        # Convert 0-based to 1-based for Julia arrays
        delta_tau[tau_indices[k] + 1, i_indices[k] + 1, j_indices[k] + 1] = real_parts[k]
    end

    return tau_grid, delta_tau
end



function gen_hyb(n_tau::Int, n_matsubara::Int, beta::Float64, n_spin_impurity::Int, n_orbital_impurity::Int, hyb_type::String; off_diagonal_prefactor::Float64 = 0.0, file_name::String = "", init_dlr_euv::Float64 = 2.0)
    tau_grid = collect(LinRange(0.0, beta, n_tau))
    n_grid = collect(0:n_matsubara)
    wn_grid = pi / beta * (2.0 .* n_grid .+ 1.0)
    if hyb_type == "SemiCircular"
        dlrGrid = DLRGrid(2.0, beta, 1e-14, true)
        G_tau = -1 * 2/pi * Sample.SemiCircle(dlrGrid, :τ, tau_grid)
        G_iwn = -1 * 2/pi * Sample.SemiCircle(dlrGrid, :n, n_grid)
        n_flavors = n_spin_impurity * n_orbital_impurity
        G_tau_tensor = zeros(Float64, n_tau, n_flavors, n_flavors)
        G_iwn_tensor = zeros(ComplexF64, n_matsubara+1, n_flavors, n_flavors)
        # Construct the G_tau and G_iwn tensors of shape (n_tau, n_falvors, n_flavors) and (n_matsubara, n_falvors, n_flavors), only diagonal elements are non-zero
        # off-diagonal elements are multiplied by off_diagonal_prefactor  for the same spin flavors, 0 otherwise
        #         upindex = (o - 1) * n_spin_impurity + 1
        #        downindex = (o - 1) * n_spin_impurity + 2
        for o1 in 1:n_orbital_impurity
            for s1 in 1:n_spin_impurity
                index1 = (o1 - 1) * n_spin_impurity + s1
                for o2 in 1:n_orbital_impurity
                    for s2 in 1:n_spin_impurity
                        index2 = (o2 - 1) * n_spin_impurity + s2
                        if index1 == index2
                            G_tau_tensor[:, index1, index2] = G_tau
                            G_iwn_tensor[:, index1, index2] = G_iwn
                        elseif s1 == s2
                            G_tau_tensor[:, index1, index2] = off_diagonal_prefactor * G_tau
                            G_iwn_tensor[:, index1, index2] = off_diagonal_prefactor * G_iwn
                        end
                    end
                end
            end
        end
        return GData(G_tau_tensor, tau_grid, G_iwn_tensor, wn_grid)
    elseif hyb_type == "from_seet" && file_name != ""
        n_flavors_expected = n_spin_impurity * n_orbital_impurity
        dlrGrid = DLRGrid(init_dlr_euv, beta, 1e-14, true)
        if endswith(file_name, ".h5")
            matsubara_indices, delta_iw = _read_delta_iw_from_h5(file_name, beta)
            n_flavors_read = size(delta_iw, 2)
            @assert n_flavors_read == n_flavors_expected "Dimension mismatch: n_spin_impurity * n_orbital_impurity = $n_flavors_expected but delta_iw has $n_flavors_read flavors"
            dlr = matfreq2dlr(dlrGrid, delta_iw, matsubara_indices)
        else
            tau_grid_in, delta_tau = _read_delta_tau_from_txt(file_name, beta)
            n_flavors_read = size(delta_tau, 2)
            @assert n_flavors_read == n_flavors_expected "Dimension mismatch: n_spin_impurity * n_orbital_impurity = $n_flavors_expected but delta_tau has $n_flavors_read flavors"
            dlr = tau2dlr(dlrGrid, delta_tau, tau_grid_in)
        end
        G_tau_tensor = dlr2tau(dlrGrid, dlr, tau_grid)
        G_iwn_tensor = dlr2matfreq(dlrGrid, dlr, n_grid)
        return GData(real(G_tau_tensor), tau_grid, G_iwn_tensor, wn_grid)
    else
        error("Unsupported hybridization type: $hyb_type")
    end
end

"""
    _clean_small_values!(arr, tol=1e-15)

Set values with absolute value less than `tol` to exactly zero.
Modifies the array in-place. Works for both real and complex arrays.
"""
function _clean_small_values!(arr::AbstractArray{<:Real}, tol::Float64 = 1e-15)
    @. arr = ifelse(abs(arr) < tol, zero(eltype(arr)), arr)
    return arr
end

function _clean_small_values!(arr::AbstractArray{<:Complex}, tol::Float64 = 1e-15)
    for i in eachindex(arr)
        re = real(arr[i])
        im = imag(arr[i])
        re = abs(re) < tol ? zero(re) : re
        im = abs(im) < tol ? zero(im) : im
        arr[i] = complex(re, im)
    end
    return arr
end

function gen_soe(beta::Float64, G_data::GData, soe_subroutine::String, n_soe::Int, dlr_eps::Float64; separate_spins::Bool = false)
    if n_soe == 0
        return Float64[], Array{Float64, 3}(undef, 0, size(G_data.G_tau, 2), size(G_data.G_tau, 3)), maximum(abs.(G_data.G_tau))
    end
    if soe_subroutine == "ESPRIT"
        if !separate_spins
            solution = solve(G_data.G_tau, G_data.tau_grid, ESPRIT(M = n_soe), LeastSquaresEquilibration(), is_Euclidian_Fermi = true)
            poles = -1.0*solution.s
            weights = solution.R
            error = _inf_error(G_data.G_tau, _reconstruct_green(poles, weights, G_data.tau_grid, beta))
        else
            n_flavors = size(G_data.G_tau, 2)
            poles_list = []
            weights_list = []
            max_error = 0.0
            for s in 1:2
                flavor_indices = collect(s:2:n_flavors)
                G_tau_spin = G_data.G_tau[:, flavor_indices, flavor_indices]
                solution = solve(G_tau_spin, G_data.tau_grid, ESPRIT(M = n_soe), LeastSquaresEquilibration(), is_Euclidian_Fermi = true)
                poles_s = -1.0*solution.s
                weights_s = solution.R
                error_s = _inf_error(G_tau_spin, _reconstruct_green(poles_s, weights_s, G_data.tau_grid, beta))
                max_error = max(max_error, error_s)
                push!(poles_list, poles_s)
                push!(weights_list, weights_s)
            end
            # restore the order, spin runs faster
            @assert length(poles_list[1]) == length(poles_list[2]) "Spin up and spin down must have the same number of poles"
            flavor_indices_up = collect(1:2:n_flavors)
            flavor_indices_dn = collect(2:2:n_flavors)
            n_poles_per_spin = length(poles_list[1])
            n_total_poles = 2 * n_poles_per_spin
            poles = zeros(Float64, n_total_poles)
            weights = zeros(Float64, n_total_poles, n_flavors, n_flavors)
            # Interleave: [up_1, dn_1, up_2, dn_2, ...]
            for i in 1:n_poles_per_spin
                # Spin-up at odd index
                poles[2*i - 1] = poles_list[1][i]
                weights[2*i - 1, flavor_indices_up, flavor_indices_up] = weights_list[1][i, :, :]
                # Spin-down at even index
                poles[2*i] = poles_list[2][i]
                weights[2*i, flavor_indices_dn, flavor_indices_dn] = weights_list[2][i, :, :]
            end
            error = max_error
        end
    elseif soe_subroutine == "ADAPOL"
        if separate_spins
            error("separate_spins is not yet implemented for ADAPOL")
        end
        poles, weights, _ = apapol_fitting(G_data.G_iwn, G_data.wn_grid, Ns = n_soe)
        weights = real.(-1.0 * weights)
        error = _inf_error(G_data.G_tau, _reconstruct_green(poles, weights, G_data.tau_grid, beta))
    elseif soe_subroutine == "DLR"
        if separate_spins
            error("separate_spins is not yet implemented for DLR")
        end
        dlr = DLRGrid(2.0, beta, dlr_eps, true)
        G_dlr = tau2dlr(dlr, G_data.G_tau, G_data.tau_grid)
        poles = dlr.ω
        weights = G_dlr
        error = _inf_error(G_data.G_tau, _reconstruct_green(poles, weights, G_data.tau_grid, beta))
    else
        error("Unsupported SOE subroutine: $soe_subroutine")
    end
    # Clean up very small numerical noise (abs < 1e-15) to exactly zero
    _clean_small_values!(weights)
    return poles, weights, error
end

function gen_remaining_hyb(tau_grid::Vector{Float64}, G_tau::Array{Float64, 3}, counterterms_contribution::Array{Float64, 3}, n_matsubara::Int, beta::Float64)
    G_remain = G_tau .- counterterms_contribution
    n_grid = collect(0:n_matsubara)
    wn_grid = pi / beta * (2.0 .* n_grid .+ 1.0)
    dlr = DLRGrid(2, beta, 1e-14, true)
    G_remain_dlr = tau2dlr(dlr, G_remain, tau_grid)
    G_remain_iwn = dlr2matfreq(dlr, G_remain_dlr, n_grid)
    return GData(G_remain, tau_grid, G_remain_iwn, wn_grid)
end

function add_counterterms_to_hopping(hopping_file::String, counterterms_energy::Vector{Float64}, counterterms_hyb::Matrix{Float64})
    # Read hopping matrix
    hopping_org = read_hopping(readdlm(hopping_file))
    n_orb = size(hopping_org, 1)
    @assert size(counterterms_hyb, 2) == n_orb "Number of orbitals in counterterms_hyb must match hopping matrix size."
    n_counterterms = length(counterterms_energy)
    hopping_new = zeros(ComplexF64, n_orb + n_counterterms, n_orb + n_counterterms)
    # Fill original hopping
    hopping_new[1:n_orb, 1:n_orb] = hopping_org
    # Fill counterterms
    for b in 1:n_counterterms
        E_b = counterterms_energy[b]
        V_b = counterterms_hyb[b, :]
        # Diagonal element
        hopping_new[n_orb + b, n_orb + b] = E_b
        # Off-diagonal elements
        for i in 1:n_orb
            hopping_new[i, n_orb + b] = V_b[i]
            hopping_new[n_orb + b, i] = conj(V_b[i])
        end
    end
    # Write new hopping matrix
    write_hopping(hopping_new, hopping_file)
end

function main()
    println("=== Generating SOE for counterterms and hybridization function ===")
    args = parse_commandline()
    print_input(args)
    G_data = gen_hyb(args["n_tau"], args["n_matsubara"], args["beta"], args["n_spin_impurity"], args["n_orbital_impurity"], args["hyb_type"], off_diagonal_prefactor = args["off_diagonal_prefactor"], file_name = args["hyb_file"], init_dlr_euv = args["init_dlr_euv"])
    poles, weights, error = gen_soe(args["beta"], G_data, args["soe_subroutine_counterterms"], args["n_soe_counterterms"], args["dlr_eps_counterterms"], separate_spins = args["separate_spins"])
    println("--- Counterterms SOE ---")
    println("Max hybridization: ", maximum(abs.(G_data.G_tau)))
    println("# of SOE terms: ", length(poles))
    println("SOE inf-norm error: ", @sprintf("%.6e", error))
    println("--- Obtaining counterterms ---")
    counterterms_energy, counterterms_hyb = _obtain_hyb_general(poles, weights, separate_spins = args["separate_spins"])
    # Clean up very small numerical noise in counterterms_hyb
    _clean_small_values!(counterterms_hyb)
    println("Shape of counterterms_energy: ", size(counterterms_energy))
    println("Shape of counterterms_hyb: ", size(counterterms_hyb))
    counterterms_contribution = _reconstruct_green_hyb(counterterms_energy, counterterms_hyb, G_data.tau_grid, args["beta"])
    counterterms_error = _inf_error(G_data.G_tau, counterterms_contribution)
    println("Fitting inf-norm error: ", @sprintf("%.6e", counterterms_error))
    if (counterterms_error - error) > 1e-12
        error("Reconstructed counterterms error does not match fitting error!")
    end
    println("--- Remaining hybridziation SOE ---")
    remaining_G_data = gen_remaining_hyb(G_data.tau_grid, G_data.G_tau, counterterms_contribution, args["n_matsubara"], args["beta"])
    if !args["do_subtraction_trick"]
        poles_remain, weights_remain, error_remain = gen_soe(args["beta"], remaining_G_data, args["soe_subroutine_hyb"], args["n_soe_hybridization"], args["dlr_eps_hybridization"], separate_spins = false)
    else
        if args["dlr_eps_hybridization"]>=args["dlr_eps_counterterms"] || args["n_soe_hybridization"]<=args["n_soe_counterterms"]
            error("For subtraction trick, dlr_eps_hybridization must be smaller than dlr_eps_counterterms and n_soe_hybridization must be larger than n_soe_counterterms")
        end
        println("Using subtraction trick for remaining hybridization SOE fitting...")
        poles_precise, weights_precise, error_precise = gen_soe(args["beta"], G_data, args["soe_subroutine_hyb"], args["n_soe_hybridization"], args["dlr_eps_hybridization"], separate_spins = false)
        # Now the remaining hybridization is the difference between the two SOE fits
        println("The number of precise SOE terms: ", length(poles_precise))
        poles_remain = vcat(poles_precise, poles)
        weights_remain = vcat(weights_precise, -1.0*weights)
        error_remain = _inf_error(remaining_G_data.G_tau, _reconstruct_green(poles_remain, weights_remain, remaining_G_data.tau_grid, args["beta"]))
    end
    println("Max remain hybridization: ", maximum(abs.(remaining_G_data.G_tau)))
    println("# of SOE terms: ", length(poles_remain))
    println("SOE inf-norm error: ", @sprintf("%.6e", error_remain))
    if !isdir(args["output_dir"])
        mkdir(args["output_dir"])
    end
    println("--- Modifying hopping matrix ---")
    add_counterterms_to_hopping(args["hopping_file"], counterterms_energy, counterterms_hyb)
    println("--- Saving remaining hybridization SOE ---")
    output_file_poles = joinpath(args["output_dir"], "hyb_poles.txt")
    output_file_weights = joinpath(args["output_dir"], "hyb_weights.txt")
    # directly save poles and weights to text files
    _write_hyb_poles(poles_remain, output_file_poles)
    _write_hyb_weight(weights_remain, output_file_weights)
    # store errors: counterterms_error, error_remain with comments
    output_file_error = joinpath(args["output_dir"], "soe_error.txt")
    open(output_file_error, "w") do io
        @printf(io, "# error_counterterms         error_remain\n")
        @printf(io, "%.18e   %.18e\n", counterterms_error, error_remain)
    end

    println("--- Finished ---")
end

main()
