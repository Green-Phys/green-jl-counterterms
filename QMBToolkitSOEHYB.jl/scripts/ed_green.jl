#!/usr/bin/env julia
# Direct ED Green's function calculation (atomic limit, no hybridization)
# Usage: julia --project=. scripts/ed_green.jl

using QMBToolkitED
using QMBToolkitDLR
using HDF5
using DelimitedFiles

#=============================================================================
# Parameters - modify these as needed
=============================================================================#
beta = 10.0                    # Inverse temperature
hopping_file = "test/data/hopping.txt"
interaction_file = "test/data/Uijkl.txt"
output_file = "sim.h5"

# DLR parameters
dlr_rtol = 1e-14
dlr_Euv = 500.0

# Linear grid size for output
n_tau_linear = 1001

#=============================================================================
# Main calculation
=============================================================================#

function main()
    println("=== ED Green's Function Calculation ===")
    println("beta = $beta")
    println("hopping_file = $hopping_file")
    println("interaction_file = $interaction_file")

    # Determine number of flavors and check spin conservation
    hopping_data = readdlm(hopping_file)
    n_flavors = Int(maximum(hopping_data[:, 1:2])) + 1
    println("n_flavors = $n_flavors")

    # Build basis (use spin conservation if applicable)
    basis = _build_basis(hopping_file, interaction_file, n_flavors)
    println("Built basis with $(length(basis.sectors)) sectors")

    # Build and solve Hamiltonian
    H, ann_ops = build_hamiltonian(basis, hopping_file, interaction_file)
    result = solve_hamiltonian(H)
    println("Exact diagonalization complete")

    # Create DLR grid
    dlr_grid = DLRGrid(dlr_Euv, beta, dlr_rtol, true)
    n_dlr = dlr_grid.size
    tau_dlr = dlr_grid.τ
    println("DLR grid size = $n_dlr")

    # Calculate Green's function directly on DLR tau grid
    green_dlr_tau = zeros(ComplexF64, n_flavors, n_flavors, length(tau_dlr))

    println("Computing Green's function...")
    for i in 1:n_flavors
        for j in 1:n_flavors
            # Use the new QMBToolkitED API that accepts custom tau grid
            green_dlr_tau[i, j, :] = calculate_green_tau(
                result, ann_ops[i], ann_ops[j], beta, tau_dlr)
        end
    end

    # Convert to DLR coefficients and interpolate to linear grid
    green_dlr_coeffs = collect(tau2dlr(dlr_grid, green_dlr_tau, tau_dlr; axis = 3))
    tau_linear = collect(LinRange(0.0, beta, n_tau_linear))
    green_linear = collect(dlr2tau(dlr_grid, green_dlr_coeffs, tau_linear; axis = 3))

    # Compute partition function
    Z = _compute_partition_function(result, beta)
    println("Partition function Z = $Z")

    # Save results
    println("Saving to $output_file...")
    h5open(output_file, "w") do file
        # Green's function on linear grid
        g_linear = create_group(file, "green/linear_grid")
        g_linear["green_tau"] = green_linear
        g_linear["tau"] = tau_linear

        # Green's function on DLR grid
        g_dlr = create_group(file, "green/dlr_grid")
        g_dlr["green_tau"] = green_dlr_tau
        g_dlr["tau"] = tau_dlr
        g_dlr["coeffs"] = green_dlr_coeffs

        # Model info
        model = create_group(file, "model")
        model["beta"] = beta
        model["n_flavors"] = n_flavors

        # Partition function
        pf = create_group(file, "partition_function")
        pf["partition_function"] = Z
    end

    # Print summary
    println("\n=== Results ===")
    println("G[1,1](τ=0) = $(green_linear[1,1,1])")
    println("G[1,1](τ=β) = $(green_linear[1,1,end])")
    println("G[1,1](0) + G[1,1](β) = $(green_linear[1,1,1] + green_linear[1,1,end])")
    println("Done!")
end

#=============================================================================
# Helper functions
=============================================================================#

"""
Build basis, using QN_NSz if spin is conserved, otherwise QN_N.
"""
function _build_basis(hopping_file, interaction_file, n_flavors)
    if n_flavors % 2 == 0 && _check_spin_conservation(hopping_file, interaction_file)
        # Spin conserved: use QN_NSz
        # Convention: flavor f = 2*orbital + spin, spin=0->sz=+1, spin=1->sz=-1
        qns = [QN_NSz(1, f % 2 == 0 ? 1 : -1) for f in 0:(n_flavors - 1)]
        println("Using QN_NSz (spin conservation detected)")
    else
        qns = [QN_N(1) for _ in 1:n_flavors]
        println("Using QN_N (particle number only)")
    end
    return Basis(qns)
end

"""
Check if the Hamiltonian conserves spin.
"""
function _check_spin_conservation(hopping_file, interaction_file)
    hopping_data = readdlm(hopping_file)

    # Check hopping terms
    for row_idx in 1:size(hopping_data, 1)
        i, j = Int(hopping_data[row_idx, 1]), Int(hopping_data[row_idx, 2])
        re, im = hopping_data[row_idx, 3], hopping_data[row_idx, 4]
        if (re != 0.0 || im != 0.0) && (i % 2) != (j % 2)
            return false
        end
    end

    # Check interaction terms
    interaction_lines = readlines(interaction_file)
    isempty(interaction_lines) && return true
    n_terms = parse(Int, strip(interaction_lines[1]))
    n_terms == 0 && return true

    for line_idx in 2:length(interaction_lines)
        line = strip(interaction_lines[line_idx])
        isempty(line) && continue
        parts = split(line)
        fl_i, fl_j, fl_k, fl_l = parse.(Int, parts[2:5])
        re, im = parse.(Float64, parts[6:7])
        if (re != 0.0 || im != 0.0)
            spin_in = (fl_i % 2) + (fl_k % 2)
            spin_out = (fl_l % 2) + (fl_j % 2)
            spin_in != spin_out && return false
        end
    end
    return true
end

"""
Compute partition function from eigenvalues.
"""
function _compute_partition_function(result, beta)
    sector_diags = result.eigenvalues.sector_diags
    min_E = minimum(minimum(evals) for (_, evals) in sector_diags if !isempty(evals); init = 0.0)
    Z_shifted = sum(sum(exp.(-beta .* (evals .- min_E)))
    for (_, evals) in sector_diags if !isempty(evals))
    return Z_shifted * exp(-beta * min_E)
end

# Run
main()
