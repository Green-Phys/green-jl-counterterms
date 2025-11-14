# Green's function diagram

function generate_green_diagrams(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64})
   
    if expansion_order == 1 
        diagrams1 = generate_green_given_topology(3, 2, nothing, 1.0, n_flavors, exponents)
        diagrams = stack([diagrams1], dims=1)
    elseif expansion_order == 2
        diagrams1 = generate_green_given_topology(5, 3, [(4,2)], -1.0, n_flavors, exponents)
        diagrams = stack([diagrams1], dims=1)
    elseif expansion_order == 3
        diagrams1 = generate_green_given_topology(7, 3, [(6,4),(5,2)],  1.0, n_flavors, exponents)
        diagrams2 = generate_green_given_topology(7, 4, [(6,3),(5,2)], -1.0, n_flavors, exponents)
        diagrams3 = generate_green_given_topology(7, 5, [(6,3),(4,2)],  1.0, n_flavors, exponents)
        diagrams4 = generate_green_given_topology(7, 4, [(6,2),(5,3)], 1.0, n_flavors, exponents)
        diagrams = stack([diagrams1, diagrams2, diagrams3, diagrams4], dims=1)
    else
        error("Unsupported expansion order for Green's function: $expansion_order")
        
    end
    return diagrams
end


function generate_green_given_topology(M::Int, annihilation_pos::Int, connected_pairs::Union{Vector{Tuple{Int,Int}},Nothing}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
    external_opts_loc = [1, annihilation_pos]
    empty_node = BackBoneNode(:empty, nothing, nothing, NoHyb())
    p = length(exponents)
    if M == 3
        # trivial case
        sign = -1* crossing_parity  #-1 comes from the definition of Green's function
        backbone = BackBone(:propagator_diagram, sign, zeros(Int, p), [empty_node, empty_node, empty_node], [BackBoneEdge([]) for _ in 1:(M-1)],external_opts_loc)
        return reshape([backbone], (1,1,1))
    end
    # the length of connected_pairs should be (M -2)/2 and the numbers in connected_pairst should enumerate from 2 to M-1
    @assert length(connected_pairs) == (M - 3) ÷ 2 "Number of connected pairs does not match M"
    @assert all(x -> x[1] > x[2], connected_pairs) "Connected pairs should have first index larger than second index"
    @assert issorted([x[1] for x in connected_pairs]; rev=true) "Connected pairs are not ordered properly"
    @assert sort(vcat([x[1] for x in connected_pairs], [x[2] for x in connected_pairs])) == setdiff(collect(2:M-1), [annihilation_pos]) "Connected pairs do not cover all internal nodes"
    
    base_nodes = Vector{BackBoneNode}(undef, M)
    base_nodes[1] = empty_node
    base_nodes[annihilation_pos] = empty_node
    base_nodes[M] = empty_node
    base_sign = -1 * crossing_parity #-1 comes from the definition of Green's function

    return _generate_diagram_combinations(:green_diagram, M, connected_pairs, n_flavors, exponents, base_nodes, base_sign, external_opts_loc)
end