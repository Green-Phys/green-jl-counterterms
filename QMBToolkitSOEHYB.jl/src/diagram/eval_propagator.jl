# Propagator diagram evaluation for bare expansion
using MPI

"""
    eval_propagator_given_order!(solver_data, diagrams, comm)

Evaluate all propagator diagrams at a given order and accumulate into solver_data.

Uses format-agnostic interface functions for:
- Creating local accumulators
- Accumulating contributions
- MPI reduction

# Arguments
- `solver_data::SolverData{T, Format}`: Solver data containing model and calculation results
- `diagrams::Array{BackBone,4}`: All diagrams for this order
- `comm::MPI.Comm`: MPI communicator for parallel evaluation

# Side Effects
- Updates propagator by adding contributions from all diagrams

# Parallelization
- Diagrams are distributed across MPI ranks in round-robin fashion
- Each rank evaluates its assigned diagrams and accumulates locally
- Results are summed across all ranks using MPI.Allreduce
- The global sum is added to the existing propagator values
"""
function eval_propagator_given_order!(solver_data::SolverData{T, Format}, diagrams::Array{BackBone,4}, comm::MPI.Comm) where {T<:Number, Format<:HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)  # Cache accessor result

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    # Create local accumulator using format-agnostic interface
    PropType = typeof(propagator.vals)
    local_result = create_local_accumulator(PropType, propagator.vals)

    # Allocate format-specific buffers
    buffers = allocate_propagator_buffers(solver_data)

    # Distribute diagrams across ranks: rank processes indices rank+1:nprocs:n_diagrams
    my_diagrams = rank+1:nprocs:n_diagrams

    # Evaluate diagrams assigned to this rank
    for i in my_diagrams
        diagram = diagrams_flat[i]
        # Format-specific backbone evaluation
        contribution = eval_propagator_backbone(diagram, solver_data, buffers)
        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
        # Use format-agnostic accumulation
        accumulate_contribution!(local_result, prefactor, contribution)
    end

    # MPI reduce and add to propagator using format-agnostic interface
    mpi_reduce_and_add!(propagator, local_result, comm)
end
