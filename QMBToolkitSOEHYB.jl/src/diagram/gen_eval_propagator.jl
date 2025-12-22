# Propagator diagram generation and evaluation for bare expansion
#
# This file contains format-independent diagram generation functions.
# Format-specific evaluation is in src/dense/propagator.jl and src/blockdense/propagator.jl

using MPI

"""
    gen_propagator_given_order(expansion_order, n_flavors, exponents)

Generate all propagator diagrams for a given expansion order.

This function is format-independent - it only generates diagram structure/topology.

# Arguments
- `expansion_order::Int`: Order of the expansion (number of hybridization line pairs)
- `n_flavors::Int`: Number of fermion flavors
- `exponents::Vector{Float64}`: SOE exponents for hybridization decomposition

# Returns
- `Array{BackBone,3}`: Array of diagrams with dimensions (topology, flavor_combination, soe_combination)

# Algorithm
1. Generate all distinct pairing topologies with crossing counts
2. For each topology, generate all flavor and SOE combinations
3. Stack all results into a single array
"""
function gen_propagator_given_order(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64})
    pairings_with_crossings = _generate_pairings(expansion_order)
    diagrams_list = Vector{Array{BackBone,3}}()
    for (pairing, crossing_count) in pairings_with_crossings
        crossing_parity = (-1.0)^crossing_count
        new_diagrams = gen_propagator_given_topology(2 + 2 * expansion_order, pairing, crossing_parity, n_flavors, exponents)
        push!(diagrams_list, new_diagrams)
    end
    diagrams = stack(diagrams_list, dims=1)
    return diagrams
end

"""
    gen_propagator_given_topology(M, connected_pairs, crossing_parity, n_flavors, exponents)

Generate all diagrams for a fixed topology (pairing structure).

This function is format-independent - it only generates diagram structure.

# Arguments
- `M::Int`: Total number of nodes (M = 2 + 2*expansion_order)
- `connected_pairs::Vector{Tuple{Int,Int}}`: Pairs of nodes connected by hybridization lines
- `crossing_parity::Float64`: Sign factor from crossing count (-1)^(crossing_count)
- `n_flavors::Int`: Number of fermion flavors
- `exponents::Vector{Float64}`: SOE exponents

# Returns
- `Array{BackBone,3}`: Diagrams for this topology

# Notes
- Nodes 1 and M are empty boundary nodes
- Internal nodes (2 to M-1) are paired according to connected_pairs
- Uses `gen_diagram_combinations` to enumerate all flavor and SOE assignments
"""
function gen_propagator_given_topology(M::Int, connected_pairs::Vector{Tuple{Int,Int}}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
    external_opts_loc = Int[]
    empty_node = BackBoneNode(:empty, -1, -1, NoHyb())

    # Validate connected_pairs structure
    @assert length(connected_pairs) == (M - 2) ÷ 2 "Number of connected pairs does not match M"
    @assert all(x -> x[1] > x[2], connected_pairs) "Connected pairs should have first index larger than second index"
    @assert issorted([x[1] for x in connected_pairs]; rev=true) "Connected pairs are not ordered properly"
    @assert sort(vcat([x[1] for x in connected_pairs], [x[2] for x in connected_pairs])) == collect(2:M-1) "Connected pairs do not cover all internal nodes"

    base_nodes = Vector{BackBoneNode}(undef, M)
    base_nodes[1] = empty_node
    base_nodes[M] = empty_node
    base_sign = crossing_parity

    return gen_diagram_combinations(:propagator_diagram, M, connected_pairs, n_flavors, exponents, base_nodes, base_sign, external_opts_loc)
end


"""
    _generate_pairings(N)

Generate all valid pairings for N pairs of points with crossing counts.

This function is format-independent - it only generates combinatorial structure.

# Arguments
- `N::Int`: Number of hybridization line pairs (expansion order)

# Returns
- `Vector{Tuple{Vector{Tuple{Int,Int}}, Int}}`: Each element is (pairing, crossing_count)
  - pairing: Vector of node pairs that are connected
  - crossing_count: Number of line crossings in this pairing

# Algorithm
Points are labeled 2, 3, ..., 2N+1 (excluding boundary nodes 1 and 2N+2).
Uses recursive enumeration (via `recurse_pairs`) to generate all valid pairings,
then counts crossings for each (via `count_pair_crossings`).
"""
function _generate_pairings(N::Int)
    # The set of points: 2, 3, ..., 2N+1
    points = collect(2:(2*N+1))

    # Generate all configurations using shared recurse_pairs function
    all_configs = recurse_pairs(points)

    # Map to (configuration, crossing_count) using shared count_pair_crossings function
    return map(config -> (config, count_pair_crossings(config)), all_configs)
end


"""
    eval_propagator_given_order!(solver_data, diagrams, comm)

Evaluate all propagator diagrams at a given order and accumulate into solver_data.

This is the format-dependent function that handles MPI distribution and accumulation.
Format-specific evaluation is delegated to `allocate_propagator_buffers` and
`eval_propagator_backbone`.

# Arguments
- `solver_data::SolverData{T, Format}`: Solver data containing model and calculation results
- `diagrams::Array{BackBone,4}`: All diagrams for this order
- `comm::MPI.Comm`: MPI communicator for parallel evaluation

# Side Effects
- Updates propagator by adding contributions from all diagrams

# Parallelization
- Diagrams are distributed across MPI ranks in round-robin fashion
- Each rank evaluates its assigned diagrams and accumulates locally
- Results are summed across all ranks using MPI.Allreduce
- The global sum is added to the existing propagator values
"""
function eval_propagator_given_order!(solver_data::SolverData{T, DenseFormat}, diagrams::Array{BackBone,4}, comm::MPI.Comm) where {T<:Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)  # Cache accessor result

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    # Create local accumulator for this rank's work (only for this order)
    local_result = zeros(T, size(propagator.vals.data))

    # Allocate format-specific buffers (dense vs blockdense)
    buffers = allocate_propagator_buffers(solver_data)

    # Distribute diagrams across ranks: rank processes indices rank+1:nprocs:n_diagrams
    my_diagrams = rank+1:nprocs:n_diagrams

    # Evaluate diagrams assigned to this rank
    for i in my_diagrams
        diagram = diagrams_flat[i]
        # Format-specific backbone evaluation
        contribution = eval_propagator_backbone(diagram, solver_data, buffers)
        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
        local_result .+= prefactor .* contribution.data
    end

    # Sum results from all ranks for this order, then ADD to existing propagator values
    order_result = MPI.Allreduce(local_result, MPI.SUM, comm)
    propagator.vals.data .+= order_result
end

function eval_propagator_given_order!(solver_data::SolverData{T, BlockDenseFormat}, diagrams::Array{BackBone,4}, comm::MPI.Comm) where {T<:Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)  # Cache accessor result

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    # Create local accumulator (BlockDensePropagator) for this rank's work
    basis = propagator.vals.basis
    flux = propagator.vals.flux
    n_tau = length(solver_data.model.dlr_grid.tau)
    local_result = zeros(BlockDensePropagator{T}, basis, flux, n_tau)

    # Allocate format-specific buffers
    buffers = allocate_propagator_buffers(solver_data)

    # Distribute diagrams across ranks
    my_diagrams = rank+1:nprocs:n_diagrams

    # Evaluate diagrams assigned to this rank
    for i in my_diagrams
        diagram = diagrams_flat[i]
        contribution = eval_propagator_backbone(diagram, solver_data, buffers)
        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
        # Use add_scaled! for efficient in-place addition
        add_scaled!(local_result, prefactor, contribution)
    end

    # Perform MPI reduction block-by-block
    # For each block in local_result, do an Allreduce and add to propagator
    for (key, local_block) in local_result.blocks
        # Reduce this block across all ranks
        global_block = MPI.Allreduce(local_block, MPI.SUM, comm)

        # Add to propagator (create block if it doesn't exist)
        if haskey(propagator.vals.blocks, key)
            propagator.vals.blocks[key] .+= global_block
        else
            propagator.vals.blocks[key] = copy(global_block)
        end
    end
end