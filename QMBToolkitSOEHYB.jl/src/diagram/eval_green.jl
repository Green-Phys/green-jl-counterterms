# Green's function diagram evaluation using batched generation
using MPI

"""
    eval_green_given_order!(solver_data, cache, is_bold, comm; batch_size=100000)

Evaluate all Green's function diagrams at a given order using hierarchical group-based iteration.

Uses flux pre-filtering to skip invalid diagrams before evaluation. Diagrams are
grouped by (topology, direction, flavor) and distributed across MPI ranks.

Uses zero-allocation MutableBackBone for diagram iteration - a single backbone
is reset and reused for each diagram, eliminating per-diagram allocation overhead.

# Arguments

  - `solver_data::SolverData{T}`: Solver data containing model and calculation results
  - `cache::DiagramConfigCache`: Pre-built configuration cache for this order
  - `is_bold::Bool`: Whether to use bold expansion
  - `comm::MPI.Comm`: MPI communicator for parallel evaluation

# Side Effects

  - Updates green by adding contributions from all diagrams

# Parallelization

  - Diagram groups are distributed across MPI ranks
  - Each rank generates and evaluates its assigned groups
  - Results are summed across all ranks using MPI.Allreduce
"""
function eval_green_given_order!(
        solver_data::SolverData{T}, cache::DiagramConfigCache, is_bold::Bool,
        comm::MPI.Comm) where {T <: Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = is_bold ? get_propagator(solver_data.calc).vals :
                 get_bare_propagator(solver_data.model).vals

    ann_opts = get_ann_opts(solver_data.model)
    cre_opts = get_cre_opts(solver_data.model)

    local_result = zeros(T, size(solver_data.calc.green.vals)...)

    # Build valid groups with flux pre-filtering
    basis = get_bare_propagator(solver_data.model).vals.basis
    valid_groups = build_valid_groups(cache, basis)

    # Allocate evaluation buffers once
    buffers = allocate_green_buffers(solver_data, cache.order)

    # Allocate single MutableBackBone for zero-allocation iteration
    mutable_bb = allocate_mutable_backbone(cache)

    # Distribute groups across MPI ranks
    n_groups = length(valid_groups)
    if n_groups == 0
        mpi_reduce_and_add!(solver_data.calc.green, local_result, comm)
        return
    end

    # Compute this rank's group range
    base_count = n_groups ÷ nprocs
    remainder = n_groups % nprocs
    if rank < remainder
        start_group = rank * (base_count + 1) + 1
        end_group = start_group + base_count
    else
        start_group = remainder * (base_count + 1) + (rank - remainder) * base_count + 1
        end_group = start_group + base_count - 1
    end
    if start_group > n_groups
        start_group = 1
        end_group = 0
    end

    # Pre-cache frequently accessed values
    K0_inverse = solver_data.model.K0_inverse
    convolution_workspace = solver_data.model.convolution_workspace

    # Process diagrams with zero-allocation loop
    for group_idx in start_group:end_group
        group = valid_groups[group_idx]

        for exp_idx in 1:group.n_exponent_combos
            # Reset and fill the mutable backbone
            reset_backbone!(mutable_bb)
            fill_backbone_from_group!(mutable_bb, cache, group, exp_idx)

            # Get split point from mutable backbone
            split_point = mutable_bb.external_opts_loc[2]

            contrib_0_to_tau = eval_green_forward_backbone(
                mutable_bb, split_point, solver_data, propagator, buffers, !is_bold
            )

            contrib_tau_to_beta_unreflected = eval_green_reverse_backbone(
                mutable_bb, split_point, solver_data, propagator, buffers, !is_bold
            )

            # Reflection: tau -> beta - tau (using pre-computed reflection matrix)
            flux = get_flux(contrib_tau_to_beta_unreflected)
            contrib_tau_to_beta = get_contrib_tau_to_beta_reflected(buffers, flux)
            do_reflection_tau2tau!(contrib_tau_to_beta, contrib_tau_to_beta_unreflected,
                convolution_workspace.reflection_matrix, convolution_workspace.dlr_workspace)

            prefactor = compute_diagram_prefactor(mutable_bb, K0_inverse)

            @inbounds for ann_index in 1:solver_data.model.n_flavors_physical
                # Apply annihilation operator: ann_opt * contrib_0_to_tau
                # Pre-filtering guarantees at least one valid (α, β) exists,
                # but not all α may be valid. Skip invalid ones.
                ann_opt = ann_opts[ann_index]
                ann_product_flux = compute_new_flux(get_flux(contrib_0_to_tau), ann_opt)

                # Skip if this α produces invalid flux (diagram may be valid for other α)
                if !haskey(buffers.ann_product_pool, ann_product_flux)
                    continue
                end

                ann_product = get_ann_product(buffers, ann_product_flux)
                fill!(ann_product, zero(T))
                multiply_op_prop!(ann_product, ann_opt, contrib_0_to_tau)

                # Multiply: contrib_tau_to_beta * ann_product
                final_product_flux = get_flux(buffers.backbone_0_to_beta_with_ann[ann_index])
                final_product = get_final_product(buffers, final_product_flux)
                fill!(final_product, zero(T))
                multiply_prop_prop!(final_product, contrib_tau_to_beta, ann_product)

                add_scaled!(buffers.backbone_0_to_beta_with_ann[ann_index], prefactor, final_product)
            end
        end
    end

    # Compute final Green's function by applying creation operators and taking trace
    cre_product = get_cre_product(buffers)
    @inbounds for ann_index in 1:solver_data.model.n_flavors_physical
        backbone_flux = get_flux(buffers.backbone_0_to_beta_with_ann[ann_index])
        for cre_index in 1:solver_data.model.n_flavors_physical
            cre_opt = cre_opts[cre_index]

            # Skip if total flux is non-zero (trace would be zero anyway)
            total_flux = compute_new_flux_reverse(backbone_flux, cre_opt)
            if !is_zero_flux(total_flux)
                continue
            end

            fill!(cre_product, zero(T))
            multiply_prop_op!(cre_product, buffers.backbone_0_to_beta_with_ann[ann_index], cre_opt)

            trace_prop!(buffers.trace_result_buffer, cre_product)
            # New layout: (ann_flavor, cre_flavor, n_tau) - tau is now the 3rd dimension
            view(local_result, ann_index, cre_index, :) .+= buffers.trace_result_buffer
        end
    end

    # Sum results from all ranks
    mpi_reduce_and_add!(solver_data.calc.green, local_result, comm)
end
