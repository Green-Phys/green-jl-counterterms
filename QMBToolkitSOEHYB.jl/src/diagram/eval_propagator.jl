# Propagator diagram evaluation using batched generation
using MPI

"""
    eval_propagator_given_order!(solver_data, cache, comm; batch_size=100000)

Evaluate all propagator diagrams at a given order using hierarchical group-based iteration.

Uses flux pre-filtering to skip invalid diagrams before evaluation. Diagrams are
grouped by (topology, direction, flavor) and distributed across MPI ranks.

Uses zero-allocation MutableBackBone for diagram iteration - a single backbone
is reset and reused for each diagram, eliminating per-diagram allocation overhead.

# Arguments

  - `solver_data::SolverData{T}`: Solver data containing model and calculation results
  - `cache::DiagramConfigCache`: Pre-built configuration cache for this order
  - `comm::MPI.Comm`: MPI communicator for parallel evaluation

# Side Effects

  - Updates propagator by adding contributions from all diagrams

# Parallelization

  - Diagram groups are distributed across MPI ranks
  - Each rank generates and evaluates its assigned groups
  - Results are summed across all ranks using MPI.Allreduce
"""
function eval_propagator_given_order!(
        solver_data::SolverData{T}, cache::DiagramConfigCache,
        comm::MPI.Comm) where {T <: Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)
    local_result = zeros_like(propagator.vals)

    # Build valid groups with flux pre-filtering
    basis = get_bare_propagator(solver_data.model).vals.basis
    valid_groups = build_valid_groups(cache, basis)

    # Allocate evaluation buffers once
    buffers = allocate_propagator_buffers(solver_data, cache.order)

    # Allocate single MutableBackBone for zero-allocation iteration
    mutable_bb = allocate_mutable_backbone(cache)

    # Distribute groups across MPI ranks
    n_groups = length(valid_groups)
    if n_groups == 0
        mpi_reduce_and_add!(propagator, local_result, comm)
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

    # Process diagrams with zero-allocation loop
    K0_inverse = solver_data.model.K0_inverse
    for group_idx in start_group:end_group
        group = valid_groups[group_idx]

        for exp_idx in 1:group.n_exponent_combos
            # Reset and fill the mutable backbone
            reset_backbone!(mutable_bb)
            fill_backbone_from_group!(mutable_bb, cache, group, exp_idx)

            # Evaluate and accumulate
            contribution = eval_propagator_forward_backbone(mutable_bb, solver_data, buffers)
            prefactor = compute_diagram_prefactor(mutable_bb, K0_inverse)
            add_scaled!(local_result, prefactor, contribution)
        end
    end

    # Sum results from all ranks
    mpi_reduce_and_add!(propagator, local_result, comm)
end
