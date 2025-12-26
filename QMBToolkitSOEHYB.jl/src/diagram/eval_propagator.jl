# Propagator diagram evaluation for bare expansion
using MPI

"""
    eval_propagator_given_order!(solver_data, diagrams, comm)

Evaluate all propagator diagrams at a given order and accumulate into solver_data.

This is the format-dependent function that handles MPI distribution and accumulation.
Format-specific evaluation is delegated to `allocate_propagator_buffers` and
`eval_propagator_backbone`.

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
function eval_propagator_given_order!(solver_data::SolverData{T, DenseFormat}, diagrams::Array{BackBone,4}, comm::MPI.Comm) where {T<:Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)  # Cache accessor result

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    # Create local accumulator for this rank's work (only for this order)
    local_result = zeros(T, size(propagator.vals.data))

    # Allocate format-specific buffers (dense vs blockdense)
    buffers = allocate_propagator_buffers(solver_data)

    # Distribute diagrams across ranks: rank processes indices rank+1:nprocs:n_diagrams
    my_diagrams = rank+1:nprocs:n_diagrams

    # Evaluate diagrams assigned to this rank
    for i in my_diagrams
        diagram = diagrams_flat[i]
        # Format-specific backbone evaluation
        contribution = eval_propagator_backbone(diagram, solver_data, buffers)
        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
        local_result .+= prefactor .* contribution.data
    end

    # Sum results from all ranks for this order, then ADD to existing propagator values
    order_result = MPI.Allreduce(local_result, MPI.SUM, comm)
    propagator.vals.data .+= order_result
end

function eval_propagator_given_order!(solver_data::SolverData{T, BlockDenseFormat}, diagrams::Array{BackBone,4}, comm::MPI.Comm) where {T<:Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = get_propagator(solver_data.calc)  # Cache accessor result

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    # Create local accumulator (BlockDensePropagator) for this rank's work
    basis = propagator.vals.basis
    flux = propagator.vals.flux
    n_tau = length(solver_data.model.dlr_grid.tau)
    local_result = zeros(BlockDensePropagator{T}, basis, flux, n_tau)

    # Allocate format-specific buffers
    buffers = allocate_propagator_buffers(solver_data)

    # Distribute diagrams across ranks
    my_diagrams = rank+1:nprocs:n_diagrams

    # Evaluate diagrams assigned to this rank
    for i in my_diagrams
        diagram = diagrams_flat[i]
        contribution = eval_propagator_backbone(diagram, solver_data, buffers)
        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
        # Use add_scaled! for efficient in-place addition
        add_scaled!(local_result, prefactor, contribution)
    end

    # Perform MPI reduction block-by-block
    # For each block in local_result, do an Allreduce and add to propagator
    for (key, local_block) in local_result.blocks
        # Reduce this block across all ranks
        global_block = MPI.Allreduce(local_block, MPI.SUM, comm)

        # Add to propagator (create block if it doesn't exist)
        if haskey(propagator.vals.blocks, key)
            propagator.vals.blocks[key] .+= global_block
        else
            propagator.vals.blocks[key] = copy(global_block)
        end
    end
end
