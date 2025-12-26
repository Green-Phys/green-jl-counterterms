# Green's function diagram generation for both bare and bold expansion

"""
    gen_green_given_order(expansion_order, n_flavors, exponents, is_bold=false)

Generate all Green's function diagrams for a given expansion order.

# Arguments
- `expansion_order::Int`: Order of the expansion
- `n_flavors::Int`: Number of fermion flavors
- `exponents::Vector{Float64}`: SOE exponents for hybridization decomposition
- `is_bold::Bool`: Whether to use bold expansion (default: false)

# Returns
- `Array{BackBone,4}`: Array of diagrams with dimensions (topology, direction, flavor_combination, soe_combination)
"""
function gen_green_given_order(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64}, is_bold::Bool=false)

    if is_bold
        if expansion_order == 1
            diagrams1 = gen_green_given_topology(3, 2, [], 1.0, n_flavors, exponents)
            diagrams = stack([diagrams1], dims=1)
        elseif expansion_order == 2
            diagrams1 = gen_green_given_topology(5, 3, [(4, 2)], -1.0, n_flavors, exponents)
            diagrams = stack([diagrams1], dims=1)
        elseif expansion_order == 3
            diagrams1 = gen_green_given_topology(7, 3, [(6, 4), (5, 2)], 1.0, n_flavors, exponents)
            diagrams2 = gen_green_given_topology(7, 4, [(6, 3), (5, 2)], -1.0, n_flavors, exponents)
            diagrams3 = gen_green_given_topology(7, 5, [(6, 3), (4, 2)], 1.0, n_flavors, exponents)
            diagrams4 = gen_green_given_topology(7, 4, [(6, 2), (5, 3)], 1.0, n_flavors, exponents)
            diagrams = stack([diagrams1, diagrams2, diagrams3, diagrams4], dims=1)
        else
            error("Unsupported expansion order for Green's function: $expansion_order")

        end
    else
        pairings_with_crossings = _generate_pairings_with_hole(expansion_order)
        diagrams_list = Vector{Array{BackBone,3}}()
        for (pairing, annihilation_pos, crossing_count) in pairings_with_crossings
            crossing_parity = (-1.0)^crossing_count
            new_diagrams = gen_green_given_topology(2 * expansion_order + 1, annihilation_pos, pairing, crossing_parity, n_flavors, exponents)
            push!(diagrams_list, new_diagrams)
        end
        diagrams = stack(diagrams_list, dims=1)
    end
    return diagrams
end


"""
    gen_green_given_topology(M, annihilation_pos, connected_pairs, crossing_parity, n_flavors, exponents)

Generate all Green's function diagrams for a fixed topology.

# Arguments
- `M::Int`: Total number of nodes
- `annihilation_pos::Int`: Position of the annihilation operator
- `connected_pairs::Vector{Tuple{Int,Int}}`: Pairs of nodes connected by hybridization lines
- `crossing_parity::Float64`: Sign factor from crossing count
- `n_flavors::Int`: Number of fermion flavors
- `exponents::Vector{Float64}`: SOE exponents

# Returns
- `Array{BackBone,3}`: Diagrams for this topology
"""
function gen_green_given_topology(M::Int, annihilation_pos::Int, connected_pairs::Vector{Tuple{Int,Int}}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
    external_opts_loc = [1, annihilation_pos]
    empty_node = BackBoneNode(:empty, -1, -1, NoHyb())
    p = length(exponents)

    # Validate connected_pairs structure
    @assert length(connected_pairs) == (M - 3) ÷ 2 "Number of connected pairs does not match M"
    @assert all(x -> x[1] > x[2], connected_pairs) "Connected pairs should have first index larger than second index"
    @assert issorted([x[1] for x in connected_pairs]; rev=true) "Connected pairs are not ordered properly"
    @assert sort(vcat([x[1] for x in connected_pairs], [x[2] for x in connected_pairs])) == setdiff(collect(2:M-1), [annihilation_pos]) "Connected pairs do not cover all internal nodes"

    base_nodes = Vector{BackBoneNode}(undef, M)
    base_nodes[1] = empty_node
    base_nodes[annihilation_pos] = empty_node
    base_nodes[M] = empty_node
    base_sign = -1 * crossing_parity #-1 comes from the definition of Green's function

    return gen_diagram_combinations(:green_diagram, M, connected_pairs, n_flavors, exponents, base_nodes, base_sign, external_opts_loc)
end


"""
    _generate_pairings_with_hole(N)

Generate all valid pairings for N pairs of points with a hole (for Green's function).

# Arguments
- `N::Int`: Number of hybridization line pairs (expansion order)

# Returns
- `Vector{Tuple{Vector{Tuple{Int,Int}}, Int, Int}}`: Each element is (pairing, annihilation_pos, crossing_count)
"""
function _generate_pairings_with_hole(N::Int)
    # Points: 2, 3, ..., 2N
    all_points = collect(2:(2*N))

    final_output = []

    for annihilation_pos in all_points
        # Remove the hole from the pool
        current_pool = filter(x -> x != annihilation_pos, all_points)

        pairings = recurse_pairs(current_pool)

        # External Contribution: Count points smaller than annihilation_pos
        # Since points are 2, 3... this is simply (annihilation_pos - 2)
        # But we use a generic count for robustness.
        external_crossings = count(p -> p < annihilation_pos, all_points)

        for pairs in pairings
            # Internal crossings (arch vs arch) using shared function
            internal_crossings = count_pair_crossings(pairs)
            total_crossings = internal_crossings + external_crossings

            push!(final_output, (pairs, annihilation_pos, total_crossings))
        end
    end

    return final_output
end
