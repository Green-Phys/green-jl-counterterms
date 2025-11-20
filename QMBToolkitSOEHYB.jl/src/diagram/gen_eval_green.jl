# Green's function diagram for both bare and bold expansion

function gen_green_given_order(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64}, is_bold::Bool=false)
    
    if is_bold
        if expansion_order == 1 
            diagrams1 = gen_green_given_topology(3, 2, nothing, 1.0, n_flavors, exponents)
            diagrams = stack([diagrams1], dims=1)
        elseif expansion_order == 2
            diagrams1 = gen_green_given_topology(5, 3, [(4,2)], -1.0, n_flavors, exponents)
            diagrams = stack([diagrams1], dims=1)
        elseif expansion_order == 3
            diagrams1 = gen_green_given_topology(7, 3, [(6,4),(5,2)],  1.0, n_flavors, exponents)
            diagrams2 = gen_green_given_topology(7, 4, [(6,3),(5,2)], -1.0, n_flavors, exponents)
            diagrams3 = gen_green_given_topology(7, 5, [(6,3),(4,2)],  1.0, n_flavors, exponents)
            diagrams4 = gen_green_given_topology(7, 4, [(6,2),(5,3)], 1.0, n_flavors, exponents)
            diagrams = stack([diagrams1, diagrams2, diagrams3, diagrams4], dims=1)
        else
            error("Unsupported expansion order for Green's function: $expansion_order")
            
        end
    else
        # if expansion_order == 1 
        #     diagrams1 = gen_green_given_topology(3, 2, nothing, 1.0, n_flavors, exponents)
        #     diagrams = stack([diagrams1], dims=1)
        # elseif expansion_order == 2
        #     diagrams1 = gen_green_given_topology(5, 3, [(4,2)], -1.0, n_flavors, exponents)
        #     diagrams2 = gen_green_given_topology(5, 4, [(3,2)], 1.0, n_flavors, exponents)
        #     diagrams3 = gen_green_given_topology(5, 2, [(4,3)], 1.0, n_flavors, exponents)
        #     diagrams = stack([diagrams1, diagrams2, diagrams3], dims=1)
        # else
        #     error("Unsupported expansion order for Green's function: $expansion_order")
            
        # end
        if expansion_order == 1 
            diagrams1 = gen_green_given_topology(3, 2, nothing, 1.0, n_flavors, exponents)
            diagrams = stack([diagrams1], dims=1)
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
    end
    return diagrams
end


function gen_green_given_topology(M::Int, annihilation_pos::Int, connected_pairs::Union{Vector{Tuple{Int,Int}},Nothing}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
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

    return gen_diagram_combinations(:green_diagram, M, connected_pairs, n_flavors, exponents, base_nodes, base_sign, external_opts_loc)
end


function eval_green_given_order!(solver_data::SolverData, diagrams::Array{BackBone,4}, is_bold::Bool=false)
    n_topology = size(diagrams, 1)
    n_directions = size(diagrams, 2)
    n_line_flavors = size(diagrams, 3)
    n_hyb_exponents = size(diagrams, 4)
    if is_bold
        propagator = solver_data.calc.propagator
    else
        propagator = solver_data.model.bare_propagator
    end
    @showprogress "Evaluating green diagrams" for i in 1:n_topology
        for j in 1:n_directions
            external_opts_loc = diagrams[i, j, 1, 1].external_opts_loc
            @assert length(external_opts_loc) == 2 && external_opts_loc[1] == 1 "External operator locations are invalid."
            backbone_0_to_beta_with_ann = [zeros(Float64, size(propagator.vals)) for i in 1:solver_data.model.n_flavors_physical]
            for k in 1:n_line_flavors
                for l in 1:n_hyb_exponents
                    diagram = diagrams[i, j, k, l]
                    # print_diagram(diagram)
                    @assert diagram.external_opts_loc == external_opts_loc "External operator locations do not match among diagrams of the same topology and direction."
                    contrib_0_to_tau = multiplication_convolution(diagram.nodes[1:external_opts_loc[2]], diagram.edges[1:external_opts_loc[2]-1], solver_data.model.dlr_grid, solver_data.model, propagator)
                    contrib_tau_to_beta = multiplication_convolution(diagram.nodes[external_opts_loc[2]:end], diagram.edges[external_opts_loc[2]:end], solver_data.model.dlr_grid, solver_data.model, propagator)
                    contrib_tau_to_beta = do_reflection_tau2tau(solver_data.model.dlr_grid, contrib_tau_to_beta)
                    prefactor = diagram.sign
                    for (idx, Kexp) in enumerate(diagram.prefactor_Kexps)
                        if Kexp != 0
                            prefactor *= solver_data.model.K0_inverse[idx]^Kexp
                        end
                    end
                    for ann_index in 1:solver_data.model.n_flavors_physical
                        backbone_0_to_beta_with_ann[ann_index] .+= prefactor .* _multiplication_33(contrib_tau_to_beta, _multiplication_23(solver_data.model.ann_opts[ann_index], contrib_0_to_tau))
                    end
                end
            end
            for ann_index in 1:solver_data.model.n_flavors_physical
                for cre_index in 1:solver_data.model.n_flavors_physical
                    solver_data.calc.green.vals[:, ann_index, cre_index] .+= _trace_3(_multiplication_32(backbone_0_to_beta_with_ann[ann_index], solver_data.model.cre_opts[cre_index])) ./ solver_data.calc.partition_function
                end
            end
        end
    end
end


function _generate_pairings_with_hole(N::Int)
    # Points: 2, 3, ..., 2N
    all_points = collect(2:(2 * N))
    
    # 1. Internal Crossings (Arch vs Arch)
    function count_internal_crossings(pairs)
        crossings = 0
        n = length(pairs)
        for i in 1:n
            for j in (i + 1):n
                # pairs[i] is the "outer" pair (larger start index)
                v_large = pairs[i][2]
                u_small = pairs[j][1]
                v_small = pairs[j][2]
                
                # Crossing: one leg of larger pair falls inside smaller pair
                if v_small < v_large < u_small
                    crossings += 1
                end
            end
        end
        return crossings
    end

    # 2. Recursive Solver (Same as before)
    function recurse_pairs(pool)
        if isempty(pool)
            return [Tuple{Int, Int}[]]
        end

        u = pool[end] # Max element
        remaining_candidates = pool[1:end-1]
        local_results = Vector{Tuple{Int, Int}}[]

        for (i, v) in enumerate(remaining_candidates)
            new_pool = deleteat!(copy(remaining_candidates), i)
            sub_paths = recurse_pairs(new_pool)
            
            for path in sub_paths
                push!(local_results, vcat([(u, v)], path))
            end
        end
        return local_results
    end

    final_output = []

    for annihilation_pos in all_points
        # Remove the hole from the pool
        current_pool = filter(x -> x != annihilation_pos, all_points)
        
        pairings = recurse_pairs(current_pool)
        
        # 3. External Contribution: Count points smaller than annihilation_pos
        # Since points are 2, 3... this is simply (annihilation_pos - 2)
        # But we use a generic count for robustness.
        external_crossings = count(p -> p < annihilation_pos, all_points)

        for pairs in pairings
            internal_crossings = count_internal_crossings(pairs)
            total_crossings = internal_crossings + external_crossings
            
            push!(final_output, (pairs, annihilation_pos, total_crossings))
        end
    end

    return final_output
end