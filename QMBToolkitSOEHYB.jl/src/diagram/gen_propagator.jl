# Propagator diagram topologies for bare expansion

function generate_propagator_diagrams(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64} = [-1.0, 1.0])
    
    diagrams = []
    if expansion_order == 0 
        diagrams = generate_propagator_given_topology(2, nothing, 1.0, n_flavors, exponents) 
    elseif expansion_order == 1 
        diagrams = generate_propagator_given_topology(4, [(3,2)],  1.0, n_flavors, exponents)
    elseif expansion_order == 2 
        diagrams1 = generate_propagator_given_topology(6, [(5,4),(3,2)],  1.0, n_flavors, exponents)
        diagrams2 = generate_propagator_given_topology(6, [(5,3),(4,2)], -1.0, n_flavors, exponents)
        diagrams3 = generate_propagator_given_topology(6, [(5,2),(4,3)],  1.0, n_flavors, exponents)
        diagrams = vcat(diagrams1, diagrams2, diagrams3)
    else
        error("Unsupported expansion order for Green's function: $expansion_order")
    end
    return diagrams
end

function generate_propagator_given_topology(M::Int, connected_pairs::Union{Vector{Tuple{Int,Int}},Nothing}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
    empty_node = BackBoneNode(:empty, nothing, nothing, NoHyb())
    diagrams = []
    p = length(exponents)
    if M == 2
        # trivial case
        sign = crossing_parity
        backbone = BackBone(:propagator_diagram, sign, zeros(Int, p), [empty_node, empty_node], [BackBoneEdge([]) for _ in 1:(M-1)])
        push!(diagrams, backbone)
        return diagrams
    end
    # the length of connected_pairs should be (M -2)/2 and the numbers in connected_pairst should enumerate from 2 to M-1
    @assert length(connected_pairs) == (M - 2) ÷ 2 "Number of connected pairs does not match M"
    @assert all(x -> x[1] > x[2], connected_pairs) "Connected pairs should have first index larger than second index"
    @assert issorted([x[1] for x in connected_pairs]; rev=true) "Connected pairs are not ordered properly"
    @assert sort(vcat([x[1] for x in connected_pairs], [x[2] for x in connected_pairs])) == collect(2:M-1) "Connected pairs do not cover all internal nodes"
    L = length(connected_pairs)
    ### enumerate 2^L possible forward or backward hybridization lines
    for hyb_directions in Iterators.product(fill((:forward, :backward), L)...)
        ### enumerate n_flavors^L possible flavor assignments (one side of the hybridization line contain bar, and the corresponding flavor index is summed over)
        for flavor_indices in Iterators.product(fill(1:n_flavors, L)...)
            ### enumerate p^L possible hybridization decompositions exponents
            for exponent_indices in Iterators.product(fill(1:p, L)...)
                nodes = Vector{BackBoneNode}(undef, M)
                edges = [BackBoneEdge([]) for _ in 1:(M-1)]
                nodes[1] = empty_node
                nodes[M] = empty_node
                prefactor_Kexps = zeros(Int, p)
                sign = crossing_parity
                for (idx, (i, j)) in enumerate(connected_pairs)
                    hyb_direction = hyb_directions[idx]
                    flavor_index = flavor_indices[idx]
                    exponent_index = exponent_indices[idx]
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
                        prefactor_Kexps[exponent_index] += i-j
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
                        prefactor_Kexps[exponent_index] += i-j
                        sign *= -1.0 # this is due to separation of backward hybridization line
                    else
                        error("Unrecognized hybridization direction or exponent sign")
                    end
                end
                push!(diagrams, BackBone(:propagator_diagram, sign, prefactor_Kexps, nodes, edges))
            end
        end
    end
    return diagrams
end