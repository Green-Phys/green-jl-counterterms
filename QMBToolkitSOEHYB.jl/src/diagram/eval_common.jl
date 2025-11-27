### Evaluate the backbone diagrams to obtain their contributions
using MPI
using LinearAlgebra
using QMBToolkitDLR

"""
    eval_diagrams!(solver_data, solver_params, diagram_type, diagrams, comm)

Evaluate all diagrams for a given diagram type and accumulate results into solver_data.

# Arguments
- `solver_data::SolverData`: The solver data containing model information and calculation results
- `solver_params::SolverParams`: Solver parameters including expansion orders and boldness settings
- `diagram_type::Symbol`: Type of diagrams to evaluate (`:propagator_diagram`, `:green_diagram`, or `:selfenergy_diagram`)
- `diagrams::Vector{Array{BackBone,4}}`: Vector of diagram arrays organized by expansion order
- `comm::MPI.Comm`: MPI communicator for parallel execution

For propagator diagrams, updates `solver_data.calc.propagator` and `solver_data.calc.partition_function`.
For Green's function diagrams, updates `solver_data.calc.green`.
"""
function eval_diagrams!(solver_data::SolverData, solver_params::SolverParams, diagram_type::Symbol, diagrams::Vector{Array{BackBone,4}}, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    if diagram_type == :propagator_diagram &&
           !solver_params.propagator.is_bold
        # Clear partition function by order tracking
        empty!(solver_data.calc.partition_function_by_order)

        for order in 0:solver_params.propagator.expansion_order
            if rank == 0
                println("Evaluating propagator diagrams at order ", order)
            end
            eval_propagator_given_order!(solver_data, diagrams[order+1], comm)

            # Calculate and store partition function after this order
            propagator_coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.propagator.vals, solver_data.model.dlr_grid.tau)
            propagator_at_beta = dlr2tau(solver_data.model.dlr_grid, propagator_coeffs, [solver_data.model.beta])[1, :, :]
            Z_current = tr(propagator_at_beta)
            push!(solver_data.calc.partition_function_by_order, Z_current)
        end

        # Update final dlr coefficients and partition function
        solver_data.calc.propagator.coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.propagator.vals, solver_data.model.dlr_grid.tau)
        # update partition function
        propagator_at_beta = dlr2tau(solver_data.model.dlr_grid, solver_data.calc.propagator.coeffs, [solver_data.model.beta])[1, :, :]
        solver_data.calc.partition_function = tr(propagator_at_beta)
    elseif diagram_type == :green_diagram
        for order in 1:solver_params.green.expansion_order
            if rank == 0
                println("Evaluating Green's function diagrams at order ", order)
            end
            eval_green_given_order!(solver_data, diagrams[order], solver_params.green.is_bold, comm)
        end
        # update dlr coefficients
        solver_data.calc.green.coeffs = tau2dlr(solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.tau)
    else
        error("Unknown diagram type: $diagram_type. Supported types are: :propagator_diagram, :green_diagram, :selfenergy_diagram")
    end
end

function multiplication_convolution(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData, propagator::DLRData)
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")
    # calculate the multiplication and convolution from right to left ignoring the two boundary nodes
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    result = copy(propagator.vals)
    for i in 1:length(edges)-1
        edge = edges[i]
        if i == 1
            for hyb in edge.attached_hybs
                result = apply_hybridization(hyb, result, model_data)
            end
        end
        # multiply with the next node operator
        node = nodes[i+1]
        if node.opt_type !== :empty
            if node.opt_type == :cdag && isnothing(node.bar_index)
                op_matrix = model_data.cre_opts[node.flavor]
            elseif node.opt_type == :c && isnothing(node.bar_index)
                op_matrix = model_data.ann_opts[node.flavor]
            elseif node.opt_type == :cdag && !isnothing(node.bar_index)
                op_matrix = model_data.cre_opts_SOE[node.flavor, node.bar_index]
            elseif node.opt_type == :c && !isnothing(node.bar_index)
                op_matrix = model_data.ann_opts_SOE[node.flavor, node.bar_index]
            end
            result = _multiplication_23(op_matrix, result)
        end

        result = apply_hybridization(node.attached_hyb, result, model_data)

        # calculate the convolution with the next edge
        edge_plus1 = edges[i+1]
        to_be_convoluted = copy(propagator.vals)
        for hyb in edge_plus1.attached_hybs
            to_be_convoluted = apply_hybridization(hyb, to_be_convoluted, model_data)
        end

        result = do_convolution_dlr2tau(dlr_grid, tau2dlr(dlr_grid, to_be_convoluted, dlr_grid.tau), tau2dlr(dlr_grid, result, dlr_grid.tau))
    end
    return result
end


function multiplication_convolution_reverse(nodes::Vector{BackBoneNode}, edges::Vector{BackBoneEdge}, dlr_grid::DLRGrid, model_data::ModelData, propagator::DLRData)
    length(nodes) == length(edges) + 1 || error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")
    # calculate the multiplication and convolution from left to right ignoring the two boundary nodes (the matrix multiplications is done in the original order)
    # |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    # nodes and edges are stored from 1 to M
    result = copy(propagator.vals)
    for i in length(edges):-1:2
        edge = edges[i]
        if i == length(edges)
            for hyb in edge.attached_hybs
                result = apply_hybridization(hyb, result, model_data)
            end
        end
        node = nodes[i]
        if node.opt_type !== :empty
            if node.opt_type == :cdag && isnothing(node.bar_index)
                op_matrix = model_data.cre_opts[node.flavor]
            elseif node.opt_type == :c && isnothing(node.bar_index)
                op_matrix = model_data.ann_opts[node.flavor]
            elseif node.opt_type == :cdag && !isnothing(node.bar_index)
                op_matrix = model_data.cre_opts_SOE[node.flavor, node.bar_index]
            elseif node.opt_type == :c && !isnothing(node.bar_index)
                op_matrix = model_data.ann_opts_SOE[node.flavor, node.bar_index]
            end
            result = _multiplication_32(result, op_matrix)
        end

        result = apply_hybridization(node.attached_hyb, result, model_data, is_reverse=true)

        # calculate the convolution with the next edge
        edge_minus1 = edges[i-1]
        to_be_convoluted = copy(propagator.vals)
        for hyb in edge_minus1.attached_hybs
            to_be_convoluted = apply_hybridization(hyb, to_be_convoluted, model_data)
        end

        result = do_convolution_dlr2tau(dlr_grid, tau2dlr(dlr_grid, result, dlr_grid.tau), tau2dlr(dlr_grid, to_be_convoluted, dlr_grid.tau))
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
    o1 == o2 || error("Matrix to be traced must be square: got dimensions $o1 x $o2")
    trace_vals = zeros(Float64, n_tau)
    for k in 1:n_tau
        trace_vals[k] = tr(A[k, :, :])
    end
    return trace_vals
end


"""
    apply_hybridization(hyb::HybridizationState, result::Array{Float64,3}, model_data::ModelData)

Apply a hybridization function multiplication to the result array.

For KPlus hybridizations, multiplies by KPlus_on_DLRtau.
For KMinus hybridizations, multiplies by KMinus_on_DLRtau.
For NoHyb or DeltaHyb, returns the result unchanged.
"""
function apply_hybridization(hyb::HybridizationState, result::Array{Float64,3}, model_data::ModelData; is_reverse::Bool=false)
    if hyb isa KPlus
        if !is_reverse
            return _multiplication_13(model_data.hyb.KPlus_on_DLRtau[:, hyb.l], result)
        else
            return _multiplication_13(model_data.hyb.KMinus_on_DLRtau[:, hyb.l], result)
        end
    elseif hyb isa KMinus
        if !is_reverse
            return _multiplication_13(model_data.hyb.KMinus_on_DLRtau[:, hyb.l], result)
        else
            return _multiplication_13(model_data.hyb.KPlus_on_DLRtau[:, hyb.l], result)
        end
    else
        return result
    end
end
