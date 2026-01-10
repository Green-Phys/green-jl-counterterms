# Propagator diagram evaluation for bare expansion
using MPI

"""
    eval_propagator_given_order!(solver_data, diagrams, comm)

Evaluate all propagator diagrams at a given order and accumulate into solver_data.

Uses format-agnostic interface functions for MPI-parallel evaluation.

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
function eval_propagator_given_order!(
        solver_data::SolverData{T, Format}, diagrams::Array{BackBone, 4},
        comm::MPI.Comm) where {T <: Number, Format <: HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    local_result = zeros_like(propagator.vals)

    buffers = allocate_propagator_buffers(solver_data)

    my_diagrams = (rank + 1):nprocs:n_diagrams

    for i in my_diagrams
        diagram = diagrams_flat[i]
        contribution = eval_propagator_forward_backbone(diagram, solver_data, buffers)
        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
        add_scaled!(local_result, prefactor, contribution)
    end

    mpi_reduce_and_add!(propagator, local_result, comm)
end
