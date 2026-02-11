#!/usr/bin/env julia
# Direct ED Green's function calculation (atomic limit, no hybridization)
# Usage: julia --project=/path/to/QMBToolkitSOEHYB.jl scripts/ed_green.jl
# MPI Usage: mpiexecjl -n <nprocs> julia --project=/path/to/QMBToolkitSOEHYB.jl scripts/ed_green.jl

using MPI
using QMBToolkitED
using QMBToolkitDLR
using HDF5
using DelimitedFiles

#=============================================================================
# Parameters - modify these as needed
=============================================================================#
beta = 10.0                    # Inverse temperature
hopping_file = "test/data_2spin/hopping.txt"
interaction_file = "test/data_2spin/Uijkl.txt"
output_file = "ed_green.h5"

# DLR parameters
dlr_rtol = 1e-14
dlr_Euv = 500.0

# Linear grid size for output
n_tau_linear = 1001

# Flavor selection parameters
# n_flavor_physical: Only calculate Green's function for flavors 1:n_flavor_physical
#                    Set to nothing to calculate all flavors
n_flavor_physical = nothing

# is_spin_conserved: If true and spin conservation is detected, skip Green's function
#                    entries that mix spin up and spin down.
#                    Assumes spin runs faster than orbital: flavor 1 = orb1 up,
#                    flavor 2 = orb1 down, flavor 3 = orb2 up, etc.
is_spin_conserved = true

#=============================================================================
# MPI Setup
=============================================================================#
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
is_root = (rank == 0)

# MPI reduce wrapper for Green's function calculation
function mpi_allreduce_sum(arr)
    result = similar(arr)
    MPI.Allreduce!(arr, result, +, comm)
    return result
end

# MPI gather wrapper for solve_hamiltonian
function mpi_gather_dicts(local_sector_diags, local_evec_blocks)
    # Gather all local dictionaries to all ranks
    # Each rank broadcasts its local data to all others

    merged_sector_diags = Dict{Int, Vector{Float64}}()
    merged_evec_blocks = Dict{Tuple{Int, Int}, Matrix{ComplexF64}}()

    for r in 0:(nprocs - 1)
        # Broadcast keys and values from rank r
        if rank == r
            keys_diags = collect(keys(local_sector_diags))
        else
            keys_diags = Int[]
        end

        # Broadcast number of diagonal items
        n_diags_arr = Int[length(keys_diags)]
        MPI.Bcast!(n_diags_arr, r, comm)
        n_diags = n_diags_arr[1]

        if rank != r
            resize!(keys_diags, n_diags)
        end

        # Broadcast keys
        if n_diags > 0
            MPI.Bcast!(keys_diags, r, comm)
        end

        # For each key, broadcast the data
        for key in keys_diags
            if rank == r
                data = local_sector_diags[key]
                len_arr = Int[length(data)]
            else
                len_arr = Int[0]
            end
            MPI.Bcast!(len_arr, r, comm)

            if rank != r
                data = zeros(Float64, len_arr[1])
            end
            MPI.Bcast!(data, r, comm)
            merged_sector_diags[key] = data
        end

        # Count evec blocks from this rank
        if rank == r
            n_evecs = length(local_evec_blocks)
        else
            n_evecs = 0
        end
        n_evecs_arr = Int[n_evecs]
        MPI.Bcast!(n_evecs_arr, r, comm)
        n_evecs = n_evecs_arr[1]

        # Broadcast evec block keys and data
        if rank == r
            evec_keys = collect(keys(local_evec_blocks))
        else
            evec_keys = Tuple{Int, Int}[]
        end

        for i in 1:n_evecs
            # Broadcast key as array of 2 ints
            if rank == r
                key_arr = Int[evec_keys[i][1], evec_keys[i][2]]
            else
                key_arr = zeros(Int, 2)
            end
            MPI.Bcast!(key_arr, r, comm)
            key = (key_arr[1], key_arr[2])

            # Broadcast dimensions
            if rank == r
                data = local_evec_blocks[key]
                dims_arr = Int[size(data, 1), size(data, 2)]
            else
                dims_arr = zeros(Int, 2)
            end
            MPI.Bcast!(dims_arr, r, comm)

            if rank != r
                data = zeros(ComplexF64, dims_arr[1], dims_arr[2])
            end
            MPI.Bcast!(data, r, comm)
            merged_evec_blocks[key] = data
        end
    end

    return merged_sector_diags, merged_evec_blocks
end

#=============================================================================
# Main calculation
=============================================================================#

function main()
    t_start = time()

    is_root && println("=== ED Green's Function Calculation ===")
    is_root && println("MPI: $nprocs processes")
    is_root && println("beta = $beta")
    is_root && println("hopping_file = $hopping_file")
    is_root && println("interaction_file = $interaction_file")

    # Determine number of flavors and check spin conservation
    hopping_data = readdlm(hopping_file)
    n_flavors = Int(maximum(hopping_data[:, 1:2])) + 1
    is_root && println("n_flavors = $n_flavors")

    # Determine how many flavors to calculate
    n_calc = isnothing(n_flavor_physical) ? n_flavors : min(n_flavor_physical, n_flavors)
    is_root && println("n_flavor_physical = $n_calc")

    # Check if spin conservation applies
    spin_conserved_detected = (n_flavors % 2 == 0) &&
                              _check_spin_conservation(hopping_file, interaction_file)
    use_spin_filter = is_spin_conserved && spin_conserved_detected
    is_root && println("is_spin_conserved = $is_spin_conserved (detected: $spin_conserved_detected, active: $use_spin_filter)")

    # Build basis (use spin conservation if applicable)
    basis = _build_basis(hopping_file, interaction_file, n_flavors)
    is_root && println("Built basis with $(length(basis.sectors)) sectors")

    # Build Hamiltonian (serial - usually fast)
    H, ann_ops = build_hamiltonian(basis, hopping_file, interaction_file)

    # Solve Hamiltonian with MPI parallelization over sectors
    parallel_info_diag = nprocs > 1 ? (rank, nprocs, mpi_gather_dicts) : nothing
    result = solve_hamiltonian(H; parallel_info = parallel_info_diag)
    is_root && println("Exact diagonalization complete")

    # Create grids
    dlr_grid = DLRGrid(dlr_Euv, beta, dlr_rtol, true)
    tau_dlr = dlr_grid.τ
    tau_linear = collect(LinRange(0.0, beta, n_tau_linear))
    is_root && println("DLR grid size = $(length(tau_dlr))")
    is_root && println("Linear grid size = $n_tau_linear")

    # Calculate Green's function
    green_dlr = zeros(ComplexF64, n_calc, n_calc, length(tau_dlr))
    green_linear = zeros(ComplexF64, n_calc, n_calc, n_tau_linear)

    # Build list of (i, j) pairs to compute
    pairs_to_compute = Tuple{Int, Int}[]
    for i in 1:n_calc
        for j in 1:n_calc
            # Skip if spin conservation is active and spins differ
            if use_spin_filter
                spin_i = (i - 1) % 2  # 0 = up, 1 = down
                spin_j = (j - 1) % 2
                if spin_i != spin_j
                    continue
                end
            end
            push!(pairs_to_compute, (i, j))
        end
    end

    is_root && println("Computing Green's function for $(length(pairs_to_compute)) (i,j) pairs...")

    # Parallel info for Green's function calculation
    parallel_info_green = nprocs > 1 ? (rank, nprocs, mpi_allreduce_sum) : nothing

    for (i, j) in pairs_to_compute
        green_dlr[i, j, :] = calculate_green_tau(
            result, ann_ops[i], ann_ops[j], beta, tau_dlr;
            parallel_info = parallel_info_green
        )
        green_linear[i, j, :] = calculate_green_tau(
            result, ann_ops[i], ann_ops[j], beta, tau_linear;
            parallel_info = parallel_info_green
        )
    end

    # Compute partition function (all ranks compute, result is identical)
    Z = _compute_partition_function(result, beta)
    is_root && println("Partition function Z = $Z")

    # Calculate total time
    total_time = time() - t_start

    # Sum CPU time across all ranks
    total_cpu_time_all = MPI.Reduce(total_time, +, 0, comm)

    # Save results (only root)
    if is_root
        println("Saving to $output_file...")
        h5open(output_file, "w") do file
            # Green's function on linear grid
            g_linear = create_group(file, "green/linear_grid")
            g_linear["green_tau"] = green_linear
            g_linear["tau"] = tau_linear

            # Green's function on DLR grid
            g_dlr = create_group(file, "green/dlr_grid")
            g_dlr["green_tau"] = green_dlr
            g_dlr["tau"] = tau_dlr

            # Model info
            model = create_group(file, "model")
            model["beta"] = beta
            model["n_flavors"] = n_flavors
            model["n_flavor_physical"] = n_calc
            model["is_spin_conserved"] = use_spin_filter

            # Partition function
            pf = create_group(file, "partition_function")
            pf["partition_function"] = Z

            # Timing
            timing = create_group(file, "timing")
            timing["total_wall_time"] = total_time
            timing["total_cpu_time"] = total_cpu_time_all
            timing["nprocs"] = nprocs
        end

        # Print summary
        println("\n=== Results ===")
        println("G[1,1](τ=0) = $(green_linear[1,1,1])")
        println("G[1,1](τ=β) = $(green_linear[1,1,end])")
        println("G[1,1](0) + G[1,1](β) = $(green_linear[1,1,1] + green_linear[1,1,end])")
        println("Wall time: $(round(total_time, digits=3)) s")
        println("Total CPU time: $(round(total_cpu_time_all, digits=3)) s")
        println("Done!")
    end

    MPI.Barrier(comm)
end

#=============================================================================
# Helper functions
=============================================================================#

"""
Build basis, using QN_NSz if spin is conserved, otherwise QN_N.
"""
function _build_basis(hopping_file, interaction_file, n_flavors)
    if n_flavors % 2 == 0 && _check_spin_conservation(hopping_file, interaction_file)
        qns = [QN_NSz(1, f % 2 == 0 ? 1 : -1) for f in 0:(n_flavors - 1)]
        is_root && println("Using QN_NSz (spin conservation detected)")
    else
        qns = [QN_N(1) for _ in 1:n_flavors]
        is_root && println("Using QN_N (particle number only)")
    end
    return Basis(qns)
end

"""
Check if the Hamiltonian conserves spin.
"""
function _check_spin_conservation(hopping_file, interaction_file)
    hopping_data = readdlm(hopping_file)

    for row_idx in 1:size(hopping_data, 1)
        i, j = Int(hopping_data[row_idx, 1]), Int(hopping_data[row_idx, 2])
        re, im = hopping_data[row_idx, 3], hopping_data[row_idx, 4]
        if (re != 0.0 || im != 0.0) && (i % 2) != (j % 2)
            return false
        end
    end

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

main()
MPI.Finalize()
