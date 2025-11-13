# Green's function diagram

function generate_green_diagrams(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64}, annihilation_index::Int, creation_index::Int)
    
    diagrams = [] 
    if expansion_order == 0 || expansion_order == 1 
        diagrams = generate_green_given_topology(3, 2, nothing, 1.0, n_flavors, exponents, annihilation_index, creation_index)
    elseif expansion_order == 2
        diagrams = generate_green_given_topology(5, 3, [(4,2)], -1.0, n_flavors, exponents, annihilation_index, creation_index)
    elseif expansion_order == 3
        diagrams1 = generate_green_given_topology(7, 3, [(6,4),(5,2)],  1.0, n_flavors, exponents, annihilation_index, creation_index)
        diagrams2 = generate_green_given_topology(7, 4, [(6,3),(5,2)], -1.0, n_flavors, exponents, annihilation_index, creation_index)
        diagrams3 = generate_green_given_topology(7, 5, [(6,3),(4,2)],  1.0, n_flavors, exponents, annihilation_index, creation_index)
        diagrams4 = generate_green_given_topology(7, 4, [(6,2),(5,3)], 1.0, n_flavors, exponents, annihilation_index, creation_index)
        diagrams = vcat(diagrams1, diagrams2, diagrams3, diagrams4)
    else
        error("Unsupported expansion order for Green's function: $expansion_order")
        
    end
    return diagrams
end


function generate_green_given_topology(M::Int, annihilation_pos::Int, connected_pairs::Union{Vector{Tuple{Int,Int}},Nothing}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64}, annihilation_index::Int, creation_index::Int)
    empty_node = BackBoneNode(:empty, nothing, nothing, NoHyb())
    creation_node = BackBoneNode(:cdag, creation_index, nothing, NoHyb())
    annihilation_node = BackBoneNode(:c, annihilation_index, nothing, NoHyb())
    diagrams = []
    p = length(exponents)
    if M == 3
        # trivial case
        sign = -1* crossing_parity  #-1 comes from the definition of Green's function
        backbone = BackBone(:propagator_diagram, sign, zeros(Int, p), [creation_node, annihilation_node, empty_node], [BackBoneEdge([]) for _ in 1:(M-1)])
        push!(diagrams, backbone)
        return diagrams
    end
    # the length of connected_pairs should be (M -2)/2 and the numbers in connected_pairst should enumerate from 2 to M-1
    @assert length(connected_pairs) == (M - 3) ÷ 2 "Number of connected pairs does not match M"
    @assert all(x -> x[1] > x[2], connected_pairs) "Connected pairs should have first index larger than second index"
    @assert issorted([x[1] for x in connected_pairs]; rev=true) "Connected pairs are not ordered properly"
    @assert sort(vcat([x[1] for x in connected_pairs], [x[2] for x in connected_pairs])) == setdiff(collect(2:M-1), [annihilation_pos]) "Connected pairs do not cover all internal nodes"
    L = length(connected_pairs)
    ### enumerate 2^L possible forward or backward hybridization lines
    for hyb_directions in Iterators.product(fill((:forward, :backward), L)...)
        ### enumerate n_flavors^L possible flavor assignments (one side of the hybridization line contain bar, and the corresponding flavor index is summed over)
        for flavor_indices in Iterators.product(fill(1:n_flavors, L)...)
            ### enumerate p^L possible hybridization decompositions exponents
            for exponent_indices in Iterators.product(fill(1:p, L)...)
                nodes = Vector{BackBoneNode}(undef, M)
                edges = [BackBoneEdge([]) for _ in 1:(M-1)]
                nodes[1] = creation_node
                nodes[annihilation_pos] = annihilation_node
                nodes[M] = empty_node
                prefactor_Kexps = zeros(Int, p)
                sign = -1* crossing_parity #-1 comes from the definition of Green's function
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