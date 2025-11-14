# Propagator diagram topologies for bare expansion
using ProgressMeter

function gen_propagator_given_order(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64})

    if expansion_order == 0
        diagrams1 = gen_propagator_given_topology(2, nothing, 1.0, n_flavors, exponents)
        diagrams = stack([diagrams1], dims=1)
    elseif expansion_order == 1
        diagrams1 = gen_propagator_given_topology(4, [(3, 2)], 1.0, n_flavors, exponents)
        diagrams = stack([diagrams1], dims=1)
    elseif expansion_order == 2
        diagrams1 = gen_propagator_given_topology(6, [(5, 4), (3, 2)], 1.0, n_flavors, exponents)
        diagrams2 = gen_propagator_given_topology(6, [(5, 3), (4, 2)], -1.0, n_flavors, exponents)
        diagrams3 = gen_propagator_given_topology(6, [(5, 2), (4, 3)], 1.0, n_flavors, exponents)
        diagrams = stack([diagrams1, diagrams2, diagrams3], dims=1)
    else
        error("Unsupported expansion order for Green's function: $expansion_order")
    end
    return diagrams
end

function gen_propagator_given_topology(M::Int, connected_pairs::Union{Vector{Tuple{Int,Int}},Nothing}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
    external_opts_loc = nothing
    empty_node = BackBoneNode(:empty, nothing, nothing, NoHyb())
    p = length(exponents)
    if M == 2
        # trivial case
        sign = crossing_parity
        backbone = BackBone(:propagator_diagram, sign, zeros(Int, p), [empty_node, empty_node], [BackBoneEdge([]) for _ in 1:(M-1)], external_opts_loc)
        return reshape([backbone], (1, 1, 1))
    end
    # the length of connected_pairs should be (M -2)/2 and the numbers in connected_pairst should enumerate from 2 to M-1
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


function eval_propagator_given_order!(solver_data::SolverData, diagrams::Array{BackBone,4})
    diagrams_flat = vcat(diagrams...)
    @showprogress "Evaluating propagator diagrams" for (i, diagram) in enumerate(diagrams_flat)
        backbone_0_to_tau = multiplication_convolution(diagram.nodes, diagram.edges, solver_data.model.dlr_grid, solver_data.model, solver_data.model.bare_propagator)
        prefactor = diagram.sign
        for (idx, Kexp) in enumerate(diagram.prefactor_Kexps)
            if Kexp != 0
                prefactor *= solver_data.model.K0_inverse[idx]^Kexp
            end
        end
        solver_data.calc.propagator.vals += prefactor .* backbone_0_to_tau
    end
end
