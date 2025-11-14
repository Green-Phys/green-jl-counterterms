# Common utility functions for diagram generation

function gen_diagrams(solver_data::SolverData, solver_params::SolverParams, diagram_type::Symbol)
    diagrams = Vector{Array{BackBone,4}}()
    if diagram_type == :propagator_diagram
        for order in 0:solver_params.expansion.expansion_order
            new_diagrams = gen_propagator_given_order(order, solver_data.model.n_flavors, solver_data.model.hyb.exponents)
            push!(diagrams, new_diagrams)
            println("Generated ", size(new_diagrams), " propagator diagrams at order ", order)
        end
    elseif diagram_type == :green_diagram
        for order in 1:solver_params.green.expansion_order
            new_diagrams = gen_green_given_order(order, solver_data.model.n_flavors, solver_data.model.hyb.exponents)
            push!(diagrams, new_diagrams)
            println("Generated ", size(new_diagrams), " Green's function diagrams at order ", order)
        end
    else
        error("Unknown diagram type: ", diagram_type)
    end
    return diagrams
end


function gen_diagram_combinations(
    diagram_type::Symbol,
    M::Int,
    connected_pairs::Vector{Tuple{Int,Int}},
    n_flavors::Int,
    exponents::Vector{Float64},
    base_nodes::Vector{BackBoneNode},
    base_sign::Float64,
    external_opts_loc::Union{Vector{Int},Nothing}=nothing
)
    p = length(exponents)
    L = length(connected_pairs)
    # diagrams has (2^L, n_flavors^L, p^L) shape
    diagrams = Array{BackBone,3}(undef, 2^L, n_flavors^L, p^L)

    ### enumerate 2^L possible forward or backward hybridization lines
    for (ind_x, hyb_directions) in enumerate(Iterators.product(fill((:forward, :backward), L)...))
        ### enumerate n_flavors^L possible flavor assignments (one side of the hybridization line contain bar, and the corresponding flavor index is summed over)
        for (ind_y, flavor_indices) in enumerate(Iterators.product(fill(1:n_flavors, L)...))
            ### enumerate p^L possible hybridization decompositions exponents
            for (ind_z, exponent_indices) in enumerate(Iterators.product(fill(1:p, L)...))
                nodes = copy(base_nodes)
                edges = [BackBoneEdge([]) for _ in 1:(M-1)]
                prefactor_Kexps = zeros(Int, p)
                sign = base_sign
                for (idx, (i, j)) in enumerate(connected_pairs)
                    hyb_direction = hyb_directions[idx]
                    flavor_index = flavor_indices[idx]
                    exponent_index = exponent_indices[idx]
                    if i-j==1 # TODO: this can be improved by attach Delta to the edge? seemly change np to n^2
                        # directly connected, no edge in between
                        if hyb_direction == :forward 
                            # cdag with bar at node i, c at node j, the edge in between has KPlus
                            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, NoHyb())
                            nodes[j] = BackBoneNode(:c, flavor_index, nothing, NoHyb())
                            # from jth edge to (i-1)th edge, attach KPlus
                            push!(edges[j].attached_hybs, KPlus(exponent_index))
                            sign *= -1.0 # diagram rule
                        else # backward
                            # c with bar at node i, cdag at node j, the edge in between has KMinus
                            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, NoHyb())
                            nodes[j] = BackBoneNode(:cdag, flavor_index, nothing, NoHyb())
                            # from jth edge to (i-1)th edge, attach KMinus
                            push!(edges[j].attached_hybs, KMinus(exponent_index))
                            sign *= -1.0 # transform from KPlus to KMinus
                        end
                    else
                        if hyb_direction == :forward && exponents[exponent_index] <= 0.0
                            # cdag with bar at node i with KPlus, c at node j with KMinus
                            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, KPlus(exponent_index))
                            nodes[j] = BackBoneNode(:c, flavor_index, nothing, KMinus(exponent_index))
                            prefactor_Kexps[exponent_index] += 1
                            sign *= -1.0 # this is because diagram rule, forward hybridization line contributes a -1 factor
                        elseif hyb_direction == :forward && exponents[exponent_index] > 0.0
                            # cdag with bar at node i, c at node j
                            nodes[i] = BackBoneNode(:cdag, flavor_index, exponent_index, NoHyb())
                            nodes[j] = BackBoneNode(:c, flavor_index, nothing, NoHyb())
                            # from jth edge to (i-1)th edge, attach KPlus
                            for e in j:(i-1)
                                push!(edges[e].attached_hybs, KPlus(exponent_index))
                            end
                            prefactor_Kexps[exponent_index] += i - j -1
                            sign *= -1.0 # this is because diagram rule, forward hybridization line contributes a -1 factor
                        elseif hyb_direction == :backward && exponents[exponent_index] > 0.0
                            # c with bar at node i with KMinus, cdag at node j with KPlus
                            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, KMinus(exponent_index))
                            nodes[j] = BackBoneNode(:cdag, flavor_index, nothing, KPlus(exponent_index))
                            prefactor_Kexps[exponent_index] += 1
                            sign *= -1.0  # this is due to separation of backward hybridization line
                        elseif hyb_direction == :backward && exponents[exponent_index] <= 0.0
                            # c with bar at node i, cdag at node j
                            nodes[i] = BackBoneNode(:c, flavor_index, exponent_index, NoHyb())
                            nodes[j] = BackBoneNode(:cdag, flavor_index, nothing, NoHyb())
                            # from jth edge to (i-1)th edge, attach KMinus
                            for e in j:(i-1)
                                push!(edges[e].attached_hybs, KMinus(exponent_index))
                            end
                            prefactor_Kexps[exponent_index] += i - j -1 
                            sign *= -1.0 # this is due to separation of backward hybridization line
                        else
                            error("Unrecognized hybridization direction or exponent sign")
                        end
                    end
                end
                diagrams[ind_x, ind_y, ind_z] = BackBone(diagram_type, sign, prefactor_Kexps, nodes, edges, external_opts_loc)
            end
        end
    end
    return diagrams
end
