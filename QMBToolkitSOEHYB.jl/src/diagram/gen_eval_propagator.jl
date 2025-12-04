# Propagator diagram for bare expansion
using MPI

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

function gen_propagator_given_topology(M::Int, connected_pairs::Vector{Tuple{Int,Int}}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
    external_opts_loc = Int[]
    empty_node = BackBoneNode(:empty, -1, -1, NoHyb())
    p = length(exponents)

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


function eval_propagator_given_order!(solver_data::SolverData, diagrams::Array{BackBone,4}, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    # Create local accumulator for this rank's work (only for this order)
    local_result = zeros(Float64, size(solver_data.calc.propagator.vals))

    # Pre-allocate temp buffers for in-place operations (reused across all diagrams)
    temp_buffer = zeros(Float64, length(solver_data.model.dlr_grid.tau), solver_data.model.n_fock, solver_data.model.n_fock)
    result_buffer = zeros(Float64, length(solver_data.model.dlr_grid.tau), solver_data.model.n_fock, solver_data.model.n_fock)
    convolution_buffer = zeros(Float64, length(solver_data.model.dlr_grid.tau), solver_data.model.n_fock, solver_data.model.n_fock)

    # Distribute diagrams across ranks: rank processes indices rank+1:nprocs:n_diagrams
    my_diagrams = rank+1:nprocs:n_diagrams

    # Evaluate diagrams assigned to this rank
    for i in my_diagrams
        diagram = diagrams_flat[i]
        backbone_0_to_tau = multiplication_convolution(diagram.nodes, diagram.edges, solver_data.model.dlr_grid, solver_data.model, solver_data.model.bare_propagator, temp_buffer, result_buffer, convolution_buffer)
        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
        local_result .+= prefactor .* backbone_0_to_tau
    end

    # Sum results from all ranks for this order, then ADD to existing propagator values
    order_result = MPI.Allreduce(local_result, MPI.SUM, comm)
    solver_data.calc.propagator.vals .+= order_result
end


function _generate_pairings(N::Int)
    # The set of points: 2, 3, ..., 2N+1
    points = collect(2:(2*N+1))

    # Generate all configurations using shared recurse_pairs function
    all_configs = recurse_pairs(points)

    # Map to (configuration, crossing_count) using shared count_pair_crossings function
    return map(config -> (config, count_pair_crossings(config)), all_configs)
end