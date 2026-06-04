# Green's function diagram evaluation using batched generation
using MPI

"""
    eval_green_given_order!(solver_data, cache, is_bold, params, comm; start_group_offset=0)

Evaluate all Green's function diagrams at a given order using hierarchical group-based iteration.

Uses flux pre-filtering to skip invalid diagrams before evaluation. Diagrams are
grouped by (topology, direction, flavor) and distributed across MPI ranks.

Uses zero-allocation MutableBackBone for diagram iteration - a single backbone
is reset and reused for each diagram, eliminating per-diagram allocation overhead.

# Arguments

  - `solver_data::SolverData{T}`: Solver data containing model and calculation results
  - `cache::DiagramConfigCache`: Pre-built configuration cache for this order
  - `is_bold::Bool`: Whether to use bold expansion
  - `params::SolverParams`: Solver parameters (supplies `checkpoint_group_interval`)
  - `comm::MPI.Comm`: MPI communicator for parallel evaluation
  - `start_group_offset::Int`: groups already accumulated before this call (>0 only
    on a within-order resume); evaluation begins at group `start_group_offset+1`

# Side Effects

  - Updates green by adding contributions from all diagrams
  - When `checkpoint_group_interval > 0`, writes a within-order checkpoint (rank 0)
    after every chunk of that many groups

# Parallelization

  - Diagram groups are distributed across MPI ranks
  - Each rank generates and evaluates its assigned groups
  - Results are summed across all ranks using MPI.Allreduce

# Within-order checkpointing

The deterministic group list is processed in consecutive prefix chunks of
`checkpoint_group_interval` groups. The per-`ann` accumulator
`backbone_0_to_beta_with_ann` and the trace phase are confined to a single chunk:
buffers are re-allocated (zero-initialized) per chunk and the trace runs per chunk,
which is exact because the trace is linear in that accumulator. Each chunk's
contribution is `MPI.Allreduce`-summed into the cumulative `green.vals`, so every
rank again holds an identical accumulator and the checkpoint needs only that plus
the integer count of groups done. `interval <= 0` is one chunk over all groups
(order-boundary checkpointing only — bit-identical to the pre-chunking behavior).
"""
function eval_green_given_order!(
        solver_data::SolverData{T}, cache::DiagramConfigCache, is_bold::Bool,
        params::SolverParams, comm::MPI.Comm;
        start_group_offset::Int = 0) where {T <: Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = is_bold ? get_propagator(solver_data.calc).vals :
                 get_bare_propagator(solver_data.model).vals

    ann_opts = get_ann_opts(solver_data.model)
    cre_opts = get_cre_opts(solver_data.model)

    # Build valid groups with flux pre-filtering
    basis = get_bare_propagator(solver_data.model).vals.basis
    valid_groups = build_valid_groups(cache, basis)
    n_groups = length(valid_groups)

    # No groups at this order: still do one (empty) collective reduce so every rank
    # advances together, then return.
    if n_groups == 0
        mpi_reduce_and_add!(solver_data.calc.green,
            zeros(T, size(solver_data.calc.green.vals)...), comm)
        return
    end

    # Backbone scratch reused across chunks; the green buffers are re-allocated per
    # chunk (below) because their cross-group accumulator must start at zero.
    mutable_bb = allocate_mutable_backbone(cache)
    K0_inverse = solver_data.model.K0_inverse
    convolution_workspace = solver_data.model.convolution_workspace

    interval = params.checkpoint_group_interval
    chunk = interval > 0 ? interval : n_groups

    g = start_group_offset
    while g < n_groups
        chunk_hi = min(g + chunk, n_groups)

        # Fresh per-chunk buffers: backbone_0_to_beta_with_ann is the cross-group
        # accumulator and allocate_green_buffers zero-initializes it, so each chunk
        # accumulates only its own groups (mirrors the per-order fresh allocation).
        buffers = allocate_green_buffers(solver_data, cache.order)
        local_result = zeros(T, size(solver_data.calc.green.vals)...)

        # Process this chunk's groups with zero-allocation loop
        for group_idx in rank_subrange(g + 1, chunk_hi, rank, nprocs)
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

        # Compute this chunk's Green contribution by applying creation operators and
        # taking the trace (linear, so summing per-chunk traces equals the full trace).
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

        # Fold this chunk into the cumulative green numerator across all ranks.
        mpi_reduce_and_add!(solver_data.calc.green, local_result, comm)
        g = chunk_hi

        # Within-order checkpoint between chunks only (the caller writes the
        # order-boundary checkpoint after this returns). `cache.order - 1` is the
        # highest FULLY completed order; `g` groups of `cache.order` are in flight.
        if g < n_groups
            rank == 0 && write_checkpoint_green!(
                solver_data, params, cache.order - 1, nprocs; in_flight_groups = g)
            _maybe_crash_group(:green, cache.order, g, comm)
        end
    end
    return
end
