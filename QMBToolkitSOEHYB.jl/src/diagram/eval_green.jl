# Green's function diagram evaluation for both bare and bold expansion
using MPI
using QMBToolkitDLR

"""
    GreenEvaluationBuffers{T, PropT}

Container for buffers used in Green's function evaluation.
Format-agnostic: works with both Dense and BlockDense propagators.

# Type Parameters
- `T`: Numeric element type
- `PropT`: Propagator type (DensePropagator{T} or BlockDensePropagator{T})
"""
struct GreenEvaluationBuffers{T<:Number, PropT}
    trace_result_buffer::Vector{Float64}
    # For dense: includes pre-allocated convolution and operator buffers
    # For blockdense: buffers are allocated dynamically based on block structure
    convolution_buffers::Any  # Format-specific buffer container
end

"""
    allocate_green_buffers(solver_data::SolverData{T, DenseFormat}) where T

Allocate evaluation buffers for dense format Green's function computation.
"""
function allocate_green_buffers(solver_data::SolverData{T, DenseFormat}) where T
    n_tau = length(solver_data.model.dlr_grid.tau)
    n_fock = solver_data.model.n_fock
    n_dlr = solver_data.model.dlr_grid.size

    # Pre-allocate buffers for forward and reverse backbone evaluation
    # Each uses a PropagatorConvolutionBuffers for the convolution operations
    conv_buffers = (
        fwd_buffers = PropagatorConvolutionBuffers{T}(n_tau, n_fock, n_dlr),
        rev_buffers = PropagatorConvolutionBuffers{T}(n_tau, n_fock, n_dlr),
        ann_product = zeros(DensePropagator{T}, n_tau, n_fock, n_fock),
        final_product = zeros(DensePropagator{T}, n_tau, n_fock, n_fock),
    )

    return GreenEvaluationBuffers{T, DensePropagator{T}}(
        zeros(Float64, n_tau),
        conv_buffers
    )
end

"""
    allocate_green_buffers(solver_data::SolverData{T, BlockDenseFormat}) where T

Allocate evaluation buffers for blockdense format Green's function computation.
BlockDense allocates convolution buffers dynamically due to flux changes.
Only the trace buffer is pre-allocated.
"""
function allocate_green_buffers(solver_data::SolverData{T, BlockDenseFormat}) where T
    n_tau = length(solver_data.model.dlr_grid.tau)

    # BlockDense: only trace buffer is pre-allocated, convolution buffers are dynamic
    return GreenEvaluationBuffers{T, BlockDensePropagator{T}}(
        zeros(Float64, n_tau),
        nothing  # No convolution buffers for BlockDense
    )
end

"""
    eval_green_given_order!(solver_data, diagrams, is_bold, comm)

Evaluate all Green's function diagrams at a given expansion order and accumulate results.
Format-agnostic implementation using interface functions.

# Arguments
- `solver_data::SolverData{T, Format}`: The solver data containing model information and calculation results
- `diagrams::Array{BackBone,4}`: 4D array of diagrams organized by (topology, direction, flavor, exponent)
- `is_bold::Bool`: Whether to use bold expansion (true) or bare expansion (false)
- `comm::MPI.Comm`: MPI communicator for parallel execution

Updates `solver_data.calc.green.vals` by accumulating contributions from all diagrams.
Uses MPI parallelization to distribute diagram evaluation across ranks.
"""
function eval_green_given_order!(solver_data::SolverData{T, Format}, diagrams::Array{BackBone,4}, is_bold::Bool, comm::MPI.Comm) where {T<:Number, Format<:HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    n_topology = size(diagrams, 1)
    n_directions = size(diagrams, 2)
    n_line_flavors = size(diagrams, 3)
    n_hyb_exponents = size(diagrams, 4)

    # Get propagator (format-agnostic accessor)
    propagator = is_bold ? get_propagator(solver_data.calc).vals : get_bare_propagator(solver_data.model).vals

    # Get operators (format-agnostic accessors)
    ann_opts = get_ann_opts(solver_data.model)
    cre_opts = get_cre_opts(solver_data.model)

    # Create local accumulator (Green's function is always stored as dense array)
    local_result = zeros(T, size(solver_data.calc.green.vals.data))

    # Get format-specific buffers
    buffers = allocate_green_buffers(solver_data)
    n_tau = length(solver_data.model.dlr_grid.tau)

    # Distribute topology loop across ranks
    my_topologies = rank+1:nprocs:n_topology

    for i in my_topologies
        for j in 1:n_directions
            external_opts_loc = diagrams[i, j, 1, 1].external_opts_loc
            @assert length(external_opts_loc) == 2 && external_opts_loc[1] == 1 "External operator locations are invalid."

            # Initialize backbone accumulators (will accumulate contributions for each ann_index)
            # Use Union type to allow both nothing and propagator
            PropT = typeof(propagator)
            backbone_0_to_beta_with_ann = Vector{Union{Nothing, PropT}}([nothing for _ in 1:solver_data.model.n_flavors_physical])

            for k in 1:n_line_flavors
                for l in 1:n_hyb_exponents
                    diagram = diagrams[i, j, k, l]
                    @assert diagram.external_opts_loc == external_opts_loc "External operator locations do not match among diagrams of the same topology and direction."

                    # Compute forward and reverse contributions using format-agnostic interface
                    contrib_0_to_tau = eval_forward_backbone(
                        diagram, external_opts_loc[2], solver_data, propagator, buffers
                    )
                    contrib_tau_to_beta = eval_reverse_backbone(
                        diagram, external_opts_loc[2], solver_data, propagator, buffers
                    )

                    # Reflection: tau -> beta - tau
                    contrib_tau_to_beta = do_reflection_tau2tau(solver_data.model.dlr_grid, contrib_tau_to_beta)

                    prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)

                    for ann_index in 1:solver_data.model.n_flavors_physical
                        # Apply annihilation operator: ann_opt * contrib_0_to_tau
                        ann_opt = ann_opts[ann_index]
                        ann_product = allocate_product_buffer(ann_opt, contrib_0_to_tau, n_tau)
                        multiply_op_prop!(ann_product, ann_opt, contrib_0_to_tau)

                        # Multiply: contrib_tau_to_beta * ann_product
                        final_product = allocate_product_buffer(contrib_tau_to_beta, ann_product, n_tau)
                        multiply_prop_prop!(final_product, contrib_tau_to_beta, ann_product)

                        # Accumulate
                        if backbone_0_to_beta_with_ann[ann_index] === nothing
                            backbone_0_to_beta_with_ann[ann_index] = prefactor * final_product
                        else
                            add_scaled!(backbone_0_to_beta_with_ann[ann_index], prefactor, final_product)
                        end
                    end
                end
            end

            # Compute final Green's function by applying creation operators and taking trace
            for ann_index in 1:solver_data.model.n_flavors_physical
                if backbone_0_to_beta_with_ann[ann_index] === nothing
                    continue
                end

                for cre_index in 1:solver_data.model.n_flavors_physical
                    cre_opt = cre_opts[cre_index]
                    cre_product = allocate_product_buffer(backbone_0_to_beta_with_ann[ann_index], cre_opt, n_tau)
                    multiply_prop_op!(cre_product, backbone_0_to_beta_with_ann[ann_index], cre_opt)

                    trace_prop!(buffers.trace_result_buffer, cre_product)
                    local_result[:, ann_index, cre_index] .+= buffers.trace_result_buffer
                end
            end
        end
    end

    # Sum results from all ranks for this order, then ADD to existing green values
    order_result = MPI.Allreduce(local_result, MPI.SUM, comm)
    solver_data.calc.green.vals.data .+= order_result
end

"""
    eval_forward_backbone(diagram, split_point, solver_data, propagator, buffers)

Evaluate forward backbone contribution from node 1 to split_point.
Format-agnostic: dispatches based on propagator type.
"""
function eval_forward_backbone(diagram::BackBone, split_point::Int, solver_data::SolverData{T, DenseFormat}, propagator::DensePropagator{T}, buffers::GreenEvaluationBuffers{T, DensePropagator{T}}) where T
    return multiplication_convolution(
        diagram.nodes[1:split_point],
        diagram.edges[1:split_point-1],
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator,
        buffers.convolution_buffers.fwd_buffers
    )
end

function eval_forward_backbone(diagram::BackBone, split_point::Int, solver_data::SolverData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::GreenEvaluationBuffers{T, BlockDensePropagator{T}}) where T
    # BlockDense: no buffers passed, allocates dynamically
    return multiplication_convolution_blockdense(
        diagram.nodes[1:split_point],
        diagram.edges[1:split_point-1],
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator
    )
end

"""
    eval_reverse_backbone(diagram, split_point, solver_data, propagator, buffers)

Evaluate reverse backbone contribution from split_point to end.
Format-agnostic: dispatches based on propagator type.
"""
function eval_reverse_backbone(diagram::BackBone, split_point::Int, solver_data::SolverData{T, DenseFormat}, propagator::DensePropagator{T}, buffers::GreenEvaluationBuffers{T, DensePropagator{T}}) where T
    return multiplication_convolution_reverse(
        diagram.nodes[split_point:end],
        diagram.edges[split_point:end],
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator,
        buffers.convolution_buffers.rev_buffers
    )
end

function eval_reverse_backbone(diagram::BackBone, split_point::Int, solver_data::SolverData{T, BlockDenseFormat}, propagator::BlockDensePropagator{T}, buffers::GreenEvaluationBuffers{T, BlockDensePropagator{T}}) where T
    # BlockDense: no buffers passed, allocates dynamically
    return multiplication_convolution_reverse_blockdense(
        diagram.nodes[split_point:end],
        diagram.edges[split_point:end],
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator
    )
end
