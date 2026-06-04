# Propagator diagram evaluation using batched generation
using MPI

"""
    eval_propagator_given_order!(solver_data, cache, params, comm; start_group_offset=0)

Evaluate all propagator diagrams at a given order using hierarchical group-based iteration.

Uses flux pre-filtering to skip invalid diagrams before evaluation. Diagrams are
grouped by (topology, direction, flavor) and distributed across MPI ranks.

Uses zero-allocation MutableBackBone for diagram iteration - a single backbone
is reset and reused for each diagram, eliminating per-diagram allocation overhead.

# Arguments

  - `solver_data::SolverData{T}`: Solver data containing model and calculation results
  - `cache::DiagramConfigCache`: Pre-built configuration cache for this order
  - `params::SolverParams`: Solver parameters (supplies `checkpoint_group_interval`)
  - `comm::MPI.Comm`: MPI communicator for parallel evaluation
  - `start_group_offset::Int`: groups already accumulated before this call (>0 only
    on a within-order resume); evaluation begins at group `start_group_offset+1`

# Side Effects

  - Updates propagator by adding contributions from all diagrams
  - When `checkpoint_group_interval > 0`, writes a within-order checkpoint (rank 0)
    after every chunk of that many groups

# Parallelization

  - Diagram groups are distributed across MPI ranks
  - Each rank generates and evaluates its assigned groups
  - Results are summed across all ranks using MPI.Allreduce

# Within-order checkpointing

The deterministic group list is processed in consecutive prefix chunks of
`checkpoint_group_interval` groups. After each chunk, an `MPI.Allreduce`-sum folds
that chunk into the cumulative propagator, so every rank again holds an identical
accumulator; the checkpoint then needs only that accumulator plus the integer count
of groups done. `interval <= 0` is one chunk over all groups (order-boundary
checkpointing only — bit-identical to the pre-chunking behavior).
"""
function eval_propagator_given_order!(
        solver_data::SolverData{T}, cache::DiagramConfigCache,
        params::SolverParams, comm::MPI.Comm;
        start_group_offset::Int = 0) where {T <: Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)

    # Build valid groups with flux pre-filtering
    basis = get_bare_propagator(solver_data.model).vals.basis
    valid_groups = build_valid_groups(cache, basis)
    n_groups = length(valid_groups)

    # No groups at this order: still do one (empty) collective reduce so every rank
    # advances together, then return.
    if n_groups == 0
        mpi_reduce_and_add!(propagator, zeros_like(propagator.vals), comm)
        return
    end

    # Scratch allocated once and reused across all chunks.
    buffers = allocate_propagator_buffers(solver_data, cache.order)
    mutable_bb = allocate_mutable_backbone(cache)
    K0_inverse = solver_data.model.K0_inverse

    interval = params.checkpoint_group_interval
    chunk = interval > 0 ? interval : n_groups

    g = start_group_offset
    while g < n_groups
        chunk_hi = min(g + chunk, n_groups)

        # Each chunk is summed independently then Allreduced into the cumulative
        # propagator, leaving every rank's accumulator identical after the chunk.
        local_result = zeros_like(propagator.vals)
        for group_idx in rank_subrange(g + 1, chunk_hi, rank, nprocs)
            group = valid_groups[group_idx]
            for exp_idx in 1:group.n_exponent_combos
                reset_backbone!(mutable_bb)
                fill_backbone_from_group!(mutable_bb, cache, group, exp_idx)
                contribution = eval_propagator_forward_backbone(mutable_bb, solver_data, buffers)
                prefactor = compute_diagram_prefactor(mutable_bb, K0_inverse)
                add_scaled!(local_result, prefactor, contribution)
            end
        end
        mpi_reduce_and_add!(propagator, local_result, comm)
        g = chunk_hi

        # Within-order checkpoint between chunks only (the caller writes the
        # order-boundary checkpoint after this returns). `cache.order - 1` is the
        # highest FULLY completed order; `g` groups of `cache.order` are in flight.
        if g < n_groups
            rank == 0 && write_checkpoint_propagator!(
                solver_data, params, cache.order - 1, nprocs; in_flight_groups = g)
            _maybe_crash_group(:propagator, cache.order, g, comm)
        end
    end
    return
end
