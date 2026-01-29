# Propagator diagram evaluation using batched generation
using MPI

"""
    eval_propagator_given_order!(solver_data, cache, comm; batch_size=100000)

Evaluate all propagator diagrams at a given order using hierarchical group-based iteration.

Uses flux pre-filtering to skip invalid diagrams before evaluation. Diagrams are
grouped by (topology, direction, flavor) and distributed across MPI ranks.

# Arguments

  - `solver_data::SolverData{T}`: Solver data containing model and calculation results
  - `cache::DiagramConfigCache`: Pre-built configuration cache for this order
  - `comm::MPI.Comm`: MPI communicator for parallel evaluation
  - `batch_size::Int`: Number of diagrams per batch (default: 100000)

# Side Effects

  - Updates propagator by adding contributions from all diagrams

# Parallelization

  - Diagram groups are distributed across MPI ranks
  - Each rank generates and evaluates its assigned groups in batches
  - Results are summed across all ranks using MPI.Allreduce
"""
function eval_propagator_given_order!(
        solver_data::SolverData{T}, cache::DiagramConfigCache,
        comm::MPI.Comm; batch_size::Int = 100000) where {T <: Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)
    local_result = zeros_like(propagator.vals)

    # Build valid groups with flux pre-filtering
    basis = get_bare_propagator(solver_data.model).vals.basis
    valid_groups = build_valid_groups(cache, basis)

    # Allocate evaluation buffers once
    buffers = allocate_propagator_buffers(solver_data, cache.order)

    # Create group-based batch iterator for this rank's portion
    iter = group_batch_iterator(cache, valid_groups, batch_size, rank, nprocs)

    # Process batches (all diagrams guaranteed to have valid flux)
    for diagrams in iter
        for diagram in diagrams
            contribution = eval_propagator_forward_backbone(diagram, solver_data, buffers)
            prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
            add_scaled!(local_result, prefactor, contribution)
        end
    end

    # Sum results from all ranks
    mpi_reduce_and_add!(propagator, local_result, comm)
end
