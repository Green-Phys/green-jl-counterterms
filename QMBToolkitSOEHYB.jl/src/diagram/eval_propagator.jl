# Propagator diagram evaluation using batched generation
using MPI

"""
    eval_propagator_given_order!(solver_data, cache, comm; batch_size=100000)

Evaluate all propagator diagrams at a given order using batched generation.

Memory-efficient version that generates and evaluates diagrams in batches,
avoiding storing all diagrams in memory simultaneously.

# Arguments

  - `solver_data::SolverData{T}`: Solver data containing model and calculation results
  - `cache::DiagramConfigCache`: Pre-built configuration cache for this order
  - `comm::MPI.Comm`: MPI communicator for parallel evaluation
  - `batch_size::Int`: Number of diagrams per batch (default: 100000)

# Side Effects

  - Updates propagator by adding contributions from all diagrams

# Parallelization

  - Diagrams are distributed across MPI ranks in contiguous chunks
  - Each rank generates and evaluates its assigned diagrams in batches
  - Results are summed across all ranks using MPI.Allreduce
"""
function eval_propagator_given_order!(
        solver_data::SolverData{T}, cache::DiagramConfigCache,
        comm::MPI.Comm; batch_size::Int = 100000) where {T <: Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)
    local_result = zeros_like(propagator.vals)

    # Allocate evaluation buffers once
    buffers = allocate_propagator_buffers(solver_data, cache.order)

    # Create batch iterator for this rank's portion
    iter = batch_iterator(cache, batch_size, rank, nprocs)

    # Process batches
    for (indices, diagrams) in iter
        for diagram in diagrams
            contribution = eval_propagator_forward_backbone(diagram, solver_data, buffers)
            # Skip if contribution is empty (invalid flux encountered - diagram contributes zero)
            if isempty(contribution.blocks)
                continue
            end
            prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
            add_scaled!(local_result, prefactor, contribution)
        end
        # Batch is discarded after this iteration, freeing memory
    end

    # Sum results from all ranks
    mpi_reduce_and_add!(propagator, local_result, comm)
end
