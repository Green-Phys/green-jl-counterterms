# Green's function diagram evaluation for both bare and bold expansion
using MPI
using QMBToolkitDLR

# allocate_green_buffers is defined in format-specific files:
# - src/format/dense/propagator.jl
# - src/format/blockdense/propagator.jl

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
function eval_green_given_order!(
        solver_data::SolverData{T, Format}, diagrams::Array{BackBone, 4}, is_bold::Bool,
        comm::MPI.Comm) where {T <: Number, Format <: HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    n_topology = size(diagrams, 1)
    n_directions = size(diagrams, 2)
    n_line_flavors = size(diagrams, 3)
    n_hyb_exponents = size(diagrams, 4)

    propagator = is_bold ? get_propagator(solver_data.calc).vals :
                 get_bare_propagator(solver_data.model).vals

    ann_opts = get_ann_opts(solver_data.model)
    cre_opts = get_cre_opts(solver_data.model)

    local_result = zeros_like(solver_data.calc.green.vals)

    buffers = allocate_green_buffers(solver_data)
    n_tau = length(solver_data.model.dlr_grid.tau)

    my_topologies = (rank + 1):nprocs:n_topology

    for i in my_topologies
        for j in 1:n_directions
            external_opts_loc = diagrams[i, j, 1, 1].external_opts_loc
            @assert length(external_opts_loc) == 2 && external_opts_loc[1] == 1 "External operator locations are invalid."

            # Initialize backbone accumulators (will accumulate contributions for each ann_index)
            # Pre-allocate with correct flux from annihilation operators
            backbone_0_to_beta_with_ann = [zeros_like_with_flux(propagator, ann_opts[idx])
                                           for idx in 1:solver_data.model.n_flavors_physical]

            for k in 1:n_line_flavors
                for l in 1:n_hyb_exponents
                    diagram = diagrams[i, j, k, l]
                    @assert diagram.external_opts_loc == external_opts_loc "External operator locations do not match among diagrams of the same topology and direction."

                    # Compute forward and reverse contributions using format-agnostic interface
                    contrib_0_to_tau = eval_green_forward_backbone(
                        diagram, external_opts_loc[2], solver_data, propagator, buffers
                    )
                    contrib_tau_to_beta = eval_green_reverse_backbone(
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
                        add_scaled!(backbone_0_to_beta_with_ann[ann_index], prefactor, final_product)
                    end
                end
            end

            # Compute final Green's function by applying creation operators and taking trace
            for ann_index in 1:solver_data.model.n_flavors_physical
                for cre_index in 1:solver_data.model.n_flavors_physical
                    cre_opt = cre_opts[cre_index]
                    cre_product = allocate_product_buffer(backbone_0_to_beta_with_ann[ann_index], cre_opt, n_tau)
                    multiply_prop_op!(cre_product, backbone_0_to_beta_with_ann[ann_index], cre_opt)

                    trace_prop!(buffers.trace_result_buffer, cre_product)
                    view(local_result, :, ann_index, cre_index) .+= buffers.trace_result_buffer
                end
            end
        end
    end

    # Sum results from all ranks for this order, then ADD to existing green values
    mpi_reduce_and_add!(solver_data.calc.green, local_result, comm)
end

# eval_green_forward_backbone and eval_green_reverse_backbone are now defined in backbone_eval.jl
# as format-agnostic unified functions.
