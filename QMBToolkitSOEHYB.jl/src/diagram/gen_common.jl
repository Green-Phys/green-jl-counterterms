# Common utility functions for diagram generation
using MPI

"""
    gen_diagrams(solver_data, solver_params, diagram_type, comm)

Generate all diagrams for a given diagram type and expansion order.

# Arguments
- `solver_data::SolverData`: The solver data containing model information
- `solver_params::SolverParams`: Solver parameters including expansion orders and boldness settings
- `diagram_type::Symbol`: Type of diagrams to generate (`:propagator_diagram` or `:green_diagram`)
- `comm::MPI.Comm`: MPI communicator for parallel execution

# Returns
- `Vector{Array{BackBone,4}}`: Vector of diagram arrays, one for each expansion order

Diagrams are organized by expansion order. For propagator diagrams, generates orders 0 through
`propagator.expansion_order`. For Green's function diagrams, generates orders 1 through
`green.expansion_order`.
"""
function gen_diagrams(solver_data::SolverData, solver_params::SolverParams, diagram_type::Symbol, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    diagrams = Vector{Array{BackBone, 4}}()
    if diagram_type == :propagator_diagram && !solver_params.propagator.is_bold
        for order in 0:solver_params.propagator.expansion_order
            new_diagrams = gen_propagator_given_order(
                order, solver_data.model.n_flavors, solver_data.model.hyb.exponents,)
            push!(diagrams, new_diagrams)
            if rank == 0
                println("Generated ", size(new_diagrams), " propagator diagrams at order ", order)
            end
        end
    elseif diagram_type == :green_diagram
        for order in 1:solver_params.green.expansion_order
            new_diagrams = gen_green_given_order(order, solver_data.model.n_flavors,
                solver_data.model.hyb.exponents, solver_params.green.is_bold,)
            push!(diagrams, new_diagrams)
            if rank == 0
                println("Generated ", size(new_diagrams), " Green's function diagrams at order ", order)
            end
        end
    else
        # Only remaining case: propagator_diagram with is_bold=true
        error("Bold propagator diagram generation not yet implemented. Set propagator.is_bold=false in the input file.")
    end
    return diagrams
end

"""
    process_connected_pair!(i, j, hyb_direction, flavor_index, exponent_index, exponents, nodes, edges, prefactor_Kexps, sign)

Process a single connected pair (i, j) and update the diagram state.

Handles both directly connected nodes (i-j==1) and separated nodes with different
hybridization configurations based on direction and exponent sign.

Returns the updated sign value.
"""
function process_connected_pair!(
        i::Int,
        j::Int,
        hyb_direction::Symbol,
        flavor_index::Int,
        exponent_index::Int,
        exponents::Vector{Float64},
        nodes::Vector{BackBoneNode},
        edges::Vector{BackBoneEdge},
        prefactor_Kexps::Vector{Int},
        sign::Float64,
)
    # Special case: directly connected nodes (no edge in between)
    # NOTE: This could potentially be optimized by attaching Delta to the edge directly,
    # which would change scaling from n^p to n^2, but requires further investigation
    if i-j==1
        if hyb_direction == :forward
            # cdag with bar at node i, c at node j, the edge in between has KPlus
            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:c, flavor_index, -1, NoHyb())
            # from jth edge to (i-1)th edge, attach KPlus
            push!(edges[j].attached_hybs, KPlus(exponent_index))
            sign *= -1.0 # diagram rule
        else # backward
            # c with bar at node i, cdag at node j, the edge in between has KMinus
            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:cdag, flavor_index, -1, NoHyb())
            # from jth edge to (i-1)th edge, attach KMinus
            push!(edges[j].attached_hybs, KMinus(exponent_index))
            sign *= -1.0 # transform from KPlus to KMinus
        end
    else
        if hyb_direction == :forward && exponents[exponent_index] <= 0.0
            # cdag with bar at node i with KPlus, c at node j with KMinus
            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, KPlus(exponent_index))
            nodes[j] = BackBoneNode(:c, flavor_index, -1, KMinus(exponent_index))
            prefactor_Kexps[exponent_index] += 1
            sign *= -1.0 # this is because diagram rule, forward hybridization line contributes a -1 factor
        elseif hyb_direction == :forward && exponents[exponent_index] > 0.0
            # cdag with bar at node i, c at node j
            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:c, flavor_index, -1, NoHyb())
            # from jth edge to (i-1)th edge, attach KPlus
            for e in j:(i - 1)
                push!(edges[e].attached_hybs, KPlus(exponent_index))
            end
            prefactor_Kexps[exponent_index] += i - j - 1
            sign *= -1.0 # this is because diagram rule, forward hybridization line contributes a -1 factor
        elseif hyb_direction == :backward && exponents[exponent_index] > 0.0
            # c with bar at node i with KMinus, cdag at node j with KPlus
            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, KMinus(exponent_index))
            nodes[j] = BackBoneNode(:cdag, flavor_index, -1, KPlus(exponent_index))
            prefactor_Kexps[exponent_index] += 1
            sign *= -1.0  # this is due to separation of backward hybridization line
        elseif hyb_direction == :backward && exponents[exponent_index] <= 0.0
            # c with bar at node i, cdag at node j
            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, NoHyb())
            nodes[j] = BackBoneNode(:cdag, flavor_index, -1, NoHyb())
            # from jth edge to (i-1)th edge, attach KMinus
            for e in j:(i - 1)
                push!(edges[e].attached_hybs, KMinus(exponent_index))
            end
            prefactor_Kexps[exponent_index] += i - j - 1
            sign *= -1.0 # this is due to separation of backward hybridization line
        else
            error("Unrecognized hybridization direction or exponent sign")
        end
    end
    return sign
end

function gen_diagram_combinations(
        diagram_type::Symbol,
        M::Int,
        connected_pairs::Vector{Tuple{Int, Int}},
        n_flavors::Int,
        exponents::Vector{Float64},
        base_nodes::Vector{BackBoneNode},
        base_sign::Float64,
        external_opts_loc::Vector{Int} = Int[],
)
    p = length(exponents)
    L = length(connected_pairs)
    # diagrams has (2^L, n_flavors^L, p^L) shape
    diagrams = Array{BackBone, 3}(undef, 2^L, n_flavors^L, p^L)

    ### enumerate 2^L possible forward or backward hybridization lines
    for (ind_x, hyb_directions) in enumerate(Iterators.product(fill((:forward, :backward), L)...))
        ### enumerate n_flavors^L possible flavor assignments (one side of the hybridization line contain bar, and the corresponding flavor index is summed over)
        for (ind_y, flavor_indices) in enumerate(Iterators.product(fill(1:n_flavors, L)...))
            ### enumerate p^L possible hybridization decompositions exponents
            for (ind_z, exponent_indices) in enumerate(Iterators.product(fill(1:p, L)...))
                nodes = copy(base_nodes)
                edges = [BackBoneEdge([]) for _ in 1:(M - 1)]
                prefactor_Kexps = zeros(Int, p)
                sign = base_sign
                for (idx, (i, j)) in enumerate(connected_pairs)
                    sign = process_connected_pair!(
                        i, j,
                        hyb_directions[idx],
                        flavor_indices[idx],
                        exponent_indices[idx],
                        exponents,
                        nodes,
                        edges,
                        prefactor_Kexps,
                        sign,
                    )
                end
                diagrams[ind_x, ind_y, ind_z] = BackBone(
                    diagram_type, sign, prefactor_Kexps, nodes, edges, external_opts_loc,)
            end
        end
    end
    return diagrams
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
    crossings = 0
    n = length(pairs)
    for i in 1:n
        for j in (i + 1):n
            # pairs[i] is the "outer" pair (larger start index)
            # pairs[j] is the "inner" pair (smaller start index)
            # We know pairs[i][1] > pairs[j][1] because the list is sorted by descending first element

            v_large = pairs[i][2]
            u_small = pairs[j][1]
            v_small = pairs[j][2]

            # Condition for crossing:
            # The endpoint of the larger "arch" (v_large) falls inside
            # the smaller "arch" (between v_small and u_small).
            if v_small < v_large < u_small
                crossings += 1
            end
        end
    end
    return crossings
end
