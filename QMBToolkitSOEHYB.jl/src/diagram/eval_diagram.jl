### Evaluate the backbone diagrams to obtain their contributions

function eval_diagram(diagram::BackBone, solver_data::SolverData)
    if diagram.diagram_type == :propagator_diagram
        result = _multiplication_convolution(diagram.nodes, diagram.edges, solver_data.model.dlr_grid, solver_data.model, solver_data.model.bare_propagator)
    end
    return result
end

function _multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData, propagator::DLRData)
    @assert length(nodes) == length(edges) + 1 "Number of nodes must be one more than number of edges."
    # calculate the multiplication and convolution from right to left ignoring the two boundary nodes
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    result = propagator.vals
    for i in 1:length(edges)-1
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                if hyb isa KPlus
                    result = _multiplication_13(model_data.hyb.KPlus_on_DLRtau[:, hyb.l], result)
                elseif hyb isa KMinus
                    result = _multiplication_13(model_data.hyb.KMinus_on_DLRtau[:, hyb.l], result)
                end
            end
        end
        # multiply with the next node operator
        node = nodes[i+1]
        if node.opt_type !== :empty
            if node.opt_type == :cdag && node.bar_index == nothing
                op_matrix = model_data.cre_opts[node.flavor]
            elseif node.opt_type == :c && node.bar_index == nothing
                op_matrix = model_data.ann_opts[node.flavor]
            elseif node.opt_type == :cdag && node.bar_index != nothing
                op_matrix = model_data.cre_opts_SOE[node.flavor, node.bar_index]
            elseif node.opt_type == :c && node.bar_index != nothing
                op_matrix = model_data.ann_opts_SOE[node.flavor, node.bar_index]
            end
        end
        result = _multiplication_23(op_matrix, result)

        if node.attached_hyb isa KPlus
            result = _multiplication_13(model_data.hyb.KPlus_on_DLRtau[:, node.attached_hyb.l], result)
        elseif node.attached_hyb isa KMinus
            result = _multiplication_13(model_data.hyb.KMinus_on_DLRtau[:, node.attached_hyb.l], result)
        end

        # calculate the convolution with the next edge
        edge_plus1 = edges[i+1]
        to_be_convoluted = propagator.vals
        for hyb in edge_plus1.attached_hybs
            if hyb isa KPlus
                to_be_convoluted = _multiplication_13(model_data.hyb.KPlus_on_DLRtau[:, hyb.l], to_be_convoluted)
            elseif hyb isa KMinus
                to_be_convoluted = _multiplication_13(model_data.hyb.KMinus_on_DLRtau[:, hyb.l], to_be_convoluted)
            end
        end

        result = do_convolution_dlr2tau(dlr_grid, tau2dlr(dlr_grid, to_be_convoluted, dlr_grid.tau), tau2dlr(dlr_grid, result, dlr_grid.tau))
    end
    return result
end

function _multiplication_23(A::Array{Float64,2}, B::Array{Float64,3})
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A * B[k, :, :]
    end
    return C
end

function _multiplication_13(A::Array{Float64,1}, B::Array{Float64,3})
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k] .* B[k, :, :]
    end
    return C
end


