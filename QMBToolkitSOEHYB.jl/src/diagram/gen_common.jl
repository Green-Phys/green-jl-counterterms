# Common utility functions for diagram generation

"""
    process_connected_pair_mutable!(i, j, hyb_direction, flavor_index, exponent_index, exponents, nodes, edges, prefactor_Kexps, sign)

Process a single connected pair (i, j) and update the MutableBackBone state.

Handles both directly connected nodes (i-j==1) and separated nodes with different
hybridization configurations based on direction and exponent sign.
Uses push_hyb! for MutableBackBoneEdge modifications.

Returns the updated sign value.
"""
function process_connected_pair_mutable!(
        i::Int,
        j::Int,
        hyb_direction::Symbol,
        flavor_index::Int,
        exponent_index::Int,
        exponents::Vector{Float64},
        nodes::Vector{BackBoneNode},
        edges::Vector{MutableBackBoneEdge},
        prefactor_Kexps::Vector{Int},
        sign::Float64
)
    if i - j == 1
        if hyb_direction == :forward
            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:c, flavor_index, -1, NoHyb())
            push_hyb!(edges[j], KPlus(exponent_index))
            sign *= -1.0
        else
            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:cdag, flavor_index, -1, NoHyb())
            push_hyb!(edges[j], KMinus(exponent_index))
            sign *= -1.0
        end
    else
        if hyb_direction == :forward && exponents[exponent_index] <= 0.0
            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, KPlus(exponent_index))
            nodes[j] = BackBoneNode(:c, flavor_index, -1, KMinus(exponent_index))
            prefactor_Kexps[exponent_index] += 1
            sign *= -1.0
        elseif hyb_direction == :forward && exponents[exponent_index] > 0.0
            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:c, flavor_index, -1, NoHyb())
            for e in j:(i - 1)
                push_hyb!(edges[e], KPlus(exponent_index))
            end
            prefactor_Kexps[exponent_index] += i - j - 1
            sign *= -1.0
        elseif hyb_direction == :backward && exponents[exponent_index] > 0.0
            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, KMinus(exponent_index))
            nodes[j] = BackBoneNode(:cdag, flavor_index, -1, KPlus(exponent_index))
            prefactor_Kexps[exponent_index] += 1
            sign *= -1.0
        elseif hyb_direction == :backward && exponents[exponent_index] <= 0.0
            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:cdag, flavor_index, -1, NoHyb())
            for e in j:(i - 1)
                push_hyb!(edges[e], KMinus(exponent_index))
            end
            prefactor_Kexps[exponent_index] += i - j - 1
            sign *= -1.0
        else
            error("Unrecognized hybridization direction or exponent sign")
        end
    end
    return sign
end

"""
    recurse_pairs(pool::Vector{Int})

Generate all valid pairings of elements in the pool.
Takes the maximum element and recursively pairs it with all other elements,
then recursively processes the remaining pool.

Returns a vector of pairing configurations, where each configuration is a vector of (larger, smaller) tuples.
"""
function recurse_pairs(pool::Vector{Int})
    if isempty(pool)
        return [Tuple{Int, Int}[]]
    end

    u = pool[end]  # Maximum element
    remaining_candidates = pool[1:(end - 1)]
    results = Vector{Tuple{Int, Int}}[]

    for (i, v) in enumerate(remaining_candidates)
        new_pool = deleteat!(copy(remaining_candidates), i)
        sub_paths = recurse_pairs(new_pool)

        for path in sub_paths
            push!(results, vcat([(u, v)], path))
        end
    end
    return results
end

"""
    count_pair_crossings(pairs::Vector{Tuple{Int,Int}})

Count the number of crossings in a list of pairs (arches).
A crossing occurs when one leg of a larger pair (outer arch) falls inside a smaller pair (inner arch).

For pairs sorted by descending first element, a crossing happens when:
v_small < v_large < u_small

where pairs[i] = (u_large, v_large) and pairs[j] = (u_small, v_small) with i < j.
"""
function count_pair_crossings(pairs::Vector{Tuple{Int, Int}})
    n = length(pairs)
    # Count crossings: for each pair (i,j) with i<j, check if v_small < v_large < u_small
    # pairs[i] is the "outer" pair (larger start index), pairs[j] is the "inner" pair
    count(
        pairs[j][2] < pairs[i][2] < pairs[j][1]
    for i in 1:n for j in (i + 1):n
    )
end

"""
    _generate_pairings(N)

Generate all valid pairings for N pairs of points with crossing counts.

# Arguments

  - `N::Int`: Number of hybridization line pairs (expansion order)

# Returns

  - `Vector{Tuple{Vector{Tuple{Int,Int}}, Int}}`: Each element is (pairing, crossing_count)

      + pairing: Vector of node pairs that are connected
      + crossing_count: Number of line crossings in this pairing

# Algorithm

Points are labeled 2, 3, ..., 2N+1 (excluding boundary nodes 1 and 2N+2).
Uses recursive enumeration (via `recurse_pairs`) to generate all valid pairings,    # The set of points: 2, 3, ..., 2N+1
then counts crossings for each (via `count_pair_crossings`).
"""
function _generate_pairings(N::Int)
    # The set of points: 2, 3, ..., 2N+1
    points = collect(2:(2 * N + 1))

    # Generate all configurations using shared recurse_pairs function
    all_configs = recurse_pairs(points)

    # Map to (configuration, crossing_count) using shared count_pair_crossings function
    return map(config -> (config, count_pair_crossings(config)), all_configs)
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
    all_points = collect(2:(2 * N))

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
