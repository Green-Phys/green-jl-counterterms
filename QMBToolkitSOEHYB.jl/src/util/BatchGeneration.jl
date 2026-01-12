#=
BatchGeneration.jl - Batched diagram generation for distributed evaluation

This module provides infrastructure for memory-efficient diagram evaluation by:
1. Generating diagrams in batches rather than all at once
2. Supporting MPI-distributed generation across ranks

Key types:
- BatchIterator: Iterator for batched diagram generation

Key functions:
- generate_diagram_batch: Generate a batch of diagrams
- batch_iterator: Create iterator for MPI-distributed batching
- compute_batch_range: Compute index range for each MPI rank
=#

#=============================================================================
# Batched Diagram Generation
=============================================================================#

"""
    generate_diagram_batch(cache::DiagramConfigCache, indices::AbstractVector{Int})

Generate a batch of diagrams for the given linear indices.

This function enables memory-efficient batched evaluation by generating
only a subset of diagrams at a time.

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `indices::AbstractVector{Int}`: Vector of linear indices (1-based) to generate

# Returns

  - `Vector{BackBone}`: Vector of generated diagrams in the same order as indices
"""
function generate_diagram_batch(cache::DiagramConfigCache, indices::AbstractVector{Int})
    diagrams = Vector{BackBone}(undef, length(indices))
    for (i, linear_idx) in enumerate(indices)
        diagrams[i] = generate_diagram_at_index(cache, linear_idx)
    end
    return diagrams
end

"""
    generate_diagram_batch(cache::DiagramConfigCache, start_idx::Int, end_idx::Int)

Generate a batch of diagrams for a contiguous range of linear indices.

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `start_idx::Int`: Start of index range (inclusive, 1-based)
  - `end_idx::Int`: End of index range (inclusive, 1-based)

# Returns

  - `Vector{BackBone}`: Vector of generated diagrams
"""
function generate_diagram_batch(cache::DiagramConfigCache, start_idx::Int, end_idx::Int)
    return generate_diagram_batch(cache, start_idx:end_idx)
end

#=============================================================================
# MPI Distribution
=============================================================================#

"""
    compute_batch_range(total::Int, rank::Int, nprocs::Int, batch_size::Int)

Compute the batch range for a given MPI rank for distributed evaluation.

Distributes diagrams in contiguous chunks across ranks, with each rank
processing its chunk in batches.

# Arguments

  - `total::Int`: Total number of diagrams
  - `rank::Int`: MPI rank (0-based)
  - `nprocs::Int`: Number of MPI processes
  - `batch_size::Int`: Size of each batch

# Returns

  - `Tuple{Int, Int}`: (start_idx, end_idx) for this rank's portion (1-based, inclusive)
    Returns (0, -1) if rank has no work
"""
function compute_batch_range(total::Int, rank::Int, nprocs::Int, batch_size::Int)
    # Distribute total diagrams across ranks in contiguous chunks
    base_count = total ÷ nprocs
    remainder = total % nprocs

    # Ranks 0..(remainder-1) get one extra diagram
    if rank < remainder
        start_idx = rank * (base_count + 1) + 1
        end_idx = start_idx + base_count
    else
        start_idx = remainder * (base_count + 1) + (rank - remainder) * base_count + 1
        end_idx = start_idx + base_count - 1
    end

    # Handle edge case where rank has no work
    if start_idx > total
        return (0, -1)
    end
    end_idx = min(end_idx, total)

    return (start_idx, end_idx)
end

#=============================================================================
# BatchIterator
=============================================================================#

"""
    BatchIterator

Iterator for batched diagram generation with optional MPI distribution.

Each MPI rank gets a contiguous portion of diagrams and iterates through
its portion in batches. For single-process usage, use `BatchIterator(cache, batch_size)`
which iterates over all diagrams.

# Fields

  - `cache::DiagramConfigCache`: The configuration cache
  - `batch_size::Int`: Number of diagrams per batch
  - `start_idx::Int`: Start index for this rank (1-based)
  - `end_idx::Int`: End index for this rank (1-based)

# Usage

```julia
# Single process - iterate over all diagrams
for (batch_indices, batch_diagrams) in BatchIterator(cache, 1000)
    # Process batch_diagrams
end

# MPI distributed - each rank iterates over its portion
iter = batch_iterator(cache, batch_size, rank, nprocs)
for (batch_indices, batch_diagrams) in iter
    # Process batch_diagrams assigned to this rank
end
```
"""
struct BatchIterator{L}
    cache::DiagramConfigCache{L}
    batch_size::Int
    start_idx::Int
    end_idx::Int
end

"""
    BatchIterator(cache::DiagramConfigCache, batch_size::Int)

Create a BatchIterator that iterates over all diagrams (single-process usage).
"""
function BatchIterator(cache::DiagramConfigCache{L}, batch_size::Int) where {L}
    total = total_diagrams(cache)
    return BatchIterator{L}(cache, batch_size, 1, total)
end

"""
    batch_iterator(cache::DiagramConfigCache, batch_size::Int, rank::Int, nprocs::Int)

Create a BatchIterator for the given MPI rank (distributed usage).

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `batch_size::Int`: Number of diagrams per batch
  - `rank::Int`: MPI rank (0-based)
  - `nprocs::Int`: Number of MPI processes
"""
function batch_iterator(cache::DiagramConfigCache{L}, batch_size::Int,
        rank::Int, nprocs::Int) where {L}
    total = total_diagrams(cache)
    start_idx, end_idx = compute_batch_range(total, rank, nprocs, batch_size)
    return BatchIterator{L}(cache, batch_size, start_idx, end_idx)
end

function Base.length(iter::BatchIterator)
    if iter.end_idx < iter.start_idx
        return 0
    end
    return cld(iter.end_idx - iter.start_idx + 1, iter.batch_size)
end

function Base.iterate(iter::BatchIterator, state::Int = iter.start_idx)
    if state > iter.end_idx || iter.end_idx < iter.start_idx
        return nothing
    end

    end_idx = min(state + iter.batch_size - 1, iter.end_idx)
    indices = state:end_idx
    diagrams = generate_diagram_batch(iter.cache, indices)

    return ((indices, diagrams), end_idx + 1)
end
