### Evaluate the backbone diagrams to obtain their contributions

function eval_diagrams!(solver_data::SolverData, solver_params::SolverParams, diagram_type::Symbol, diagrams::Vector{Array{BackBone,4}})
    if diagram_type == :propagator_diagram
        @assert length(diagrams) == solver_params.expansion.expansion_order + 1 "Number of diagram sets does not match expansion order."
        for order in 0:solver_params.expansion.expansion_order
            println("Evaluating propagator diagrams at order ", order)
            eval_propagator_given_order!(solver_data, diagrams[order+1])
        end
        # update dlr coefficients
        solver_data.calc.propagator.coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.propagator.vals, solver_data.model.dlr_grid.tau)
        # update partition function
        propagator_at_beta = dlr2tau(solver_data.model.dlr_grid, solver_data.calc.propagator.coeffs, [solver_data.model.beta])[1, :, :]
        solver_data.calc.partition_function = tr(propagator_at_beta)
    elseif diagram_type == :green_diagram
        @assert length(diagrams) == solver_params.green.expansion_order "Number of diagram sets does not match green expansion order."
        for order in 1:solver_params.green.expansion_order
            println("Evaluating Green's function diagrams at order ", order)
            eval_green_given_order!(solver_data, diagrams[order])
        end
        # update dlr coefficients
        solver_data.calc.green.coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.tau)
    else
        error("Unknown diagram type: ", diagram_type)
    end
end

function multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData, propagator::DLRData)
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

function _multiplication_32(A::Array{Float64,3}, B::Array{Float64,2})
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k, :, :] * B
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

function _multiplication_33(A::Array{Float64,3}, B::Array{Float64,3})
    n_tau = size(B, 1)
    o1 = size(B, 2)
    o2 = size(B, 3)
    C = zeros(Float64, n_tau, o1, o2)
    for k in 1:n_tau
        C[k, :, :] = A[k, :, :] * B[k, :, :]
    end
    return C
end


function _trace_3(A::Array{Float64,3})
    n_tau = size(A, 1)
    o1 = size(A, 2)
    o2 = size(A, 3)
    @assert o1 == o2 "Matrix to be traced must be square."
    trace_vals = zeros(Float64, n_tau)
    for k in 1:n_tau
        trace_vals[k] = tr(A[k, :, :])  
    end
    return trace_vals
end
