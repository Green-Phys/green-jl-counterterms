using ArgParse
using Printf
using LinearAlgebra
using DelimitedFiles

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
                            i, j, k, val)
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
    data_out[:, 1] = repeat(0:n_orb-1, inner=n_orb)
    
    # j (orb_2, fastest changing index -> Col index)
    # repeat([0, 1], outer=2) -> [0, 1, 0, 1]
    data_out[:, 2] = repeat(0:n_orb-1, outer=n_orb)

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

function _obtain_hyb(poles, weights; svdtol=1e-7)
    # From the output (poles, weights) to (counterterms_energy, counterterms_hyb)
    # poles shape: (p,)
    # weights shape: (p, o1, o2)
    # counterterms_energy [E]: (Nb,)
    # counterterms_hyb [V]: (Nb, o)
    
    # Delta_{ij}(tau) = \sum_{k} weights_{ijk} K(poles_{k}, tau) = \sum_{b} -1.0 * V_{bi}^{*} V_{bj} K(E_{b}, tau)
    counterterms_energy = []
    counterterms_hyb = []

    if length(poles) == 0
        return Float64[], Array{Float64,2}(undef, 0, size(weights, 2))
    end

    # Loop over each matrix in the 'weights' list
    for i in 1:size(weights, 1)
        E = poles[i]
        sum_b_V_bi_V_bj = -1.0* weights[i, :, :]
        
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

function _reconstruct_green(poles::Vector{Float64}, weights::Array{Float64,3}, tau_grid::Vector{Float64}, beta::Float64)
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

function _inf_error(green::Array{Float64,3}, reconstructed_green::Array{Float64,3})
    return maximum(abs.(green .- reconstructed_green))
end

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
        "--soe_subroutine_hybridization"
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
    end
    return parse_args(s)
end

function print_input(args)
    println("--- Input parameters ---")
    @printf("Number of spin flavors for impurity: %d\n", args["n_spin_impurity"])
    @printf("Number of orbitals for impurity: %d\n", args["n_orbital_impurity"])
    @printf("Type of hybridization function: %s\n", args["hyb_type"])
    @printf("Hopping file to be modified: %s\n", args["hopping_file"])
    @printf("SOE subroutine for counterterms: %s\n", args["soe_subroutine_counterterms"])
    @printf("SOE subroutine for hybridization: %s\n", args["soe_subroutine_hybridization"])
    @printf("Output directory: %s\n", args["output_dir"])
end

struct GData
    G_tau::Array{Float64,3}
    tau_grid::Vector{Float64}
    G_iwn::Array{ComplexF64,3}
    wn_grid::Vector{Float64}
end

function gen_hyb(n_tau::Int, n_matsubara::Int, beta::Float64, n_spin_impurity::Int, n_orbital_impurity::Int, hyb_type::String)
    tau_grid = collect(LinRange(0.0, beta, n_tau))
    n_grid = collect(0:n_matsubara)
    wn_grid  = pi / beta * (2.0 .* n_grid .+ 1.0)
    if hyb_type == "SemiCircular"
        dlr = DLRGrid(2.0, beta, 1e-14, true)
        G_tau = -1* 2/pi * Sample.SemiCircle(dlr, :τ, tau_grid)
        G_iwn = -1* 2/pi * Sample.SemiCircle(dlr, :n, n_grid)
    else
        error("Unsupported hybridization type: $hyb_type")
    end
    n_flavors = n_spin_impurity * n_orbital_impurity
    G_tau_tensor = zeros(Float64, n_tau, n_flavors, n_flavors)
    G_iwn_tensor = zeros(ComplexF64, n_matsubara+1, n_flavors, n_flavors)
    # Construct the G_tau and G_iwn tensors of shape (n_tau, n_falvors, n_flavors) and (n_matsubara, n_falvors, n_flavors), only diagonal elements are non-zero
    for f in 1:n_flavors
        G_tau_tensor[:, f, f] = G_tau
        G_iwn_tensor[:, f, f] = G_iwn
    end
    return GData(G_tau_tensor, tau_grid, G_iwn_tensor, wn_grid)
end

function gen_soe(beta::Float64, G_data::GData, soe_subroutine::String, n_soe::Int, dlr_eps::Float64)
    if n_soe == 0
        return Float64[], Array{Float64,3}(undef, 0, size(G_data.G_tau, 2), size(G_data.G_tau, 3)), maximum(abs.(G_data.G_tau))
    end
    if soe_subroutine == "ESPRIT"
        solution = solve(G_data.G_tau, G_data.tau_grid, ESPRIT(M=n_soe), LeastSquaresEquilibration(), is_Euclidian_Fermi=true)
        poles = -1.0*solution.s
        weights = solution.R
        error = _inf_error(G_data.G_tau, _reconstruct_green(poles, weights, G_data.tau_grid, beta))
    elseif soe_subroutine == "ADAPOL"
        poles, weights, _ = apapol_fitting(G_data.G_iwn, G_data.wn_grid, Ns=n_soe)
        weights = real.(-1.0 * weights)
        error = _inf_error(G_data.G_tau, _reconstruct_green(poles, weights, G_data.tau_grid, beta))
    elseif soe_subroutine == "DLR"
        dlr = DLRGrid(2.0, beta, dlr_eps, true)
        G_dlr = tau2dlr(dlr, G_data.G_tau, G_data.tau_grid)
        poles = dlr.omega
        weights = G_dlr
        error = _inf_error(G_data.G_tau, _reconstruct_green(poles, weights, G_data.tau_grid, beta))
    else
        error("Unsupported SOE subroutine: $soe_subroutine")
    end
    return poles, weights, error
end

function gen_remaining_hyb(tau_grid::Vector{Float64}, G_tau::Array{Float64,3}, counterterms_contribution::Array{Float64,3}, n_matsubara::Int, beta::Float64)
    G_remain = G_tau .- counterterms_contribution
    n_grid = collect(0:n_matsubara)
    wn_grid  = pi / beta * (2.0 .* n_grid .+ 1.0)
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
    G_data = gen_hyb(args["n_tau"], args["n_matsubara"], args["beta"], args["n_spin_impurity"], args["n_orbital_impurity"], args["hyb_type"])
    poles, weights, error = gen_soe(args["beta"], G_data, args["soe_subroutine_counterterms"], args["n_soe_counterterms"], args["dlr_eps_counterterms"])
    println("--- Counterterms SOE ---")
    println("Max hybridization: ", maximum(abs.(G_data.G_tau)))
    println("# of SOE terms: ", length(poles))
    println("SOE inf-norm error: ", @sprintf("%.6e", error))
    println("--- Counterterms fitting ---")
    counterterms_energy, counterterms_hyb = _obtain_hyb(poles, weights)
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
    poles_remain, weights_remain, error_remain = gen_soe(args["beta"], remaining_G_data, args["soe_subroutine_hybridization"], args["n_soe_hybridization"], args["dlr_eps_hybridization"])
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
