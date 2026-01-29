# Hierarchical Diagram Iteration Implementation Plan

## Overview

Restructure diagram iteration from flat linear indexing to hierarchical group-based iteration with flux pre-filtering. This eliminates invalid diagrams before evaluation and removes the need for empty propagator sentinel buffers.

## Current Architecture

```
Linear Index → (topo, dir, flav, exp) → Generate Diagram → Evaluate → Check if empty → Accumulate
                                                              ↓
                                                    (Skip if empty blocks)
```

**Problems:**
1. Invalid flux sequences discovered AFTER expensive backbone evaluation
2. Empty propagator buffers needed as sentinels for invalid fluxes
3. Multiple `isempty(blocks)` and `haskey` checks scattered in code
4. Poor cache locality (random access pattern)

## New Architecture

```
Build Valid Groups → Distribute to MPI Ranks → For each Group → Generate all exp combinations → Evaluate → Accumulate
      ↓
(Pre-filter by flux)
```

**Benefits:**
1. Invalid flux sequences filtered BEFORE any evaluation
2. No empty propagator buffers needed
3. No runtime validity checks needed
4. Better cache locality (same operators reused within group)

---

## Implementation Phases

### Phase 1: Add Flux Validation Infrastructure

**Files to modify:** `src/diagram/DiagramConfig.jl`

**New types and functions:**

```julia
"""
    DiagramGroup

A group of diagrams sharing the same (topology, direction, flavor) configuration.
All diagrams in a group have the same flux sequence and operator structure.
"""
struct DiagramGroup
    topo_idx::Int
    dir_idx::Int
    flav_idx::Int
    n_exponent_combos::Int  # How many diagrams in this group
end

"""
    check_flux_sequence(topo_config, directions, flavors, basis) -> Bool

Check if the operator sequence produces only valid intermediate fluxes.
Returns true if all intermediate fluxes have non-empty block structure.
"""
function check_flux_sequence(
    topo_config::TopologyConfig,
    directions::NTuple{L,Symbol},
    flavors::NTuple{L,Int},
    basis::Basis
) where L
    # Implementation: simulate flux evolution through operator sequence
end

"""
    build_valid_groups(cache::DiagramConfigCache, basis::Basis) -> Vector{DiagramGroup}

Build list of valid diagram groups by pre-filtering invalid flux sequences.
"""
function build_valid_groups(cache::DiagramConfigCache, basis::Basis)
    # Implementation: iterate (topo, dir, flav), filter by check_flux_sequence
end

"""
    total_diagrams_in_groups(groups::Vector{DiagramGroup}) -> Int

Total number of diagrams across all valid groups.
"""
function total_diagrams_in_groups(groups::Vector{DiagramGroup})
    return sum(g.n_exponent_combos for g in groups)
end
```

**Testing:**
- Unit test `check_flux_sequence` with known valid/invalid combinations
- Verify `build_valid_groups` produces correct count
- Compare with current approach (count of non-empty evaluations should match)

---

### Phase 2: Add Hierarchical Batch Iterator

**Files to modify:** `src/util/BatchGeneration.jl`

**New types and functions:**

```julia
"""
    GroupBatchIterator

Iterator that processes diagram groups, generating all exponent combinations
for each valid group.
"""
struct GroupBatchIterator{L}
    cache::DiagramConfigCache{L}
    groups::Vector{DiagramGroup}
    batch_size::Int
    start_group_idx::Int
    end_group_idx::Int
end

"""
    group_batch_iterator(cache, groups, batch_size, rank, nprocs)

Create iterator for MPI-distributed group-based iteration.
Distributes groups (not individual diagrams) across ranks.
"""
function group_batch_iterator(
    cache::DiagramConfigCache{L},
    groups::Vector{DiagramGroup},
    batch_size::Int,
    rank::Int,
    nprocs::Int
) where L
    # Distribute groups across ranks
end

# Iterator interface
function Base.iterate(iter::GroupBatchIterator, state=...)
    # For current group, generate batch of diagrams across exponent combinations
    # When group exhausted, move to next group
end
```

**Design decisions:**
- Batch across exponent combinations within a group (cache locality)
- When batch spans group boundary, complete current group first
- Return both group info and diagrams for caller context

---

### Phase 3: Update Evaluation Functions

**Files to modify:**
- `src/diagram/eval_propagator.jl`
- `src/diagram/eval_green.jl`

**Changes to `eval_propagator_given_order!`:**

```julia
function eval_propagator_given_order!(
    solver_data::SolverData{T}, cache::DiagramConfigCache,
    comm::MPI.Comm; batch_size::Int = 100000
) where {T <: Number}

    # NEW: Build valid groups with flux pre-filtering
    basis = get_bare_propagator(solver_data.model).vals.basis
    valid_groups = build_valid_groups(cache, basis)

    # Allocate buffers (simplified - no empty propagators needed)
    buffers = allocate_propagator_buffers(solver_data, cache.order)

    # NEW: Use group-based iterator
    iter = group_batch_iterator(cache, valid_groups, batch_size, rank, nprocs)

    for (group, diagrams) in iter
        for diagram in diagrams
            contribution = eval_propagator_forward_backbone(diagram, solver_data, buffers)
            # REMOVED: isempty(contribution.blocks) check - guaranteed valid
            prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
            add_scaled!(local_result, prefactor, contribution)
        end
    end

    mpi_reduce_and_add!(propagator, local_result, comm)
end
```

**Similar changes to `eval_green_given_order!`**

---

### Phase 4: Simplify Backbone Evaluation

**Files to modify:** `src/diagram/backbone_eval.jl`

**Remove early exit checks:**

```julia
function multiplication_convolution(...)
    # ...

    @inbounds for i in 1:(length(edges) - 1)
        # ...

        if node.opt_type !== :empty
            op = get_operator(node, model_data)
            new_flux = compute_new_flux(current_flux, op)

            # REMOVED: Early exit check - flux guaranteed valid
            # if !haskey(buffers.op_product_pool, new_flux)
            #     return buffers.empty_propagator_tau
            # end

            op_product = get_op_product(buffers, new_flux)  # Direct access, guaranteed to exist
            # ...
        end
        # ...
    end

    return result
end
```

**Same for `multiplication_convolution_reverse`**

---

### Phase 5: Remove Empty Propagator Infrastructure

**Files to modify:** `src/format/backbone_buffers.jl`

**Changes to `PropagatorEvalBuffers`:**

```julia
struct PropagatorEvalBuffers{T <: Number, QN <: QuantumNumber}
    basis::Basis
    n_tau::Int
    n_dlr::Int

    # Variable-flux pools
    result_pool::Dict{QN, Propagator{T}}
    op_product_pool::Dict{QN, Propagator{T}}
    result_dlr_pool::Dict{QN, Propagator{T}}

    # REMOVED: empty_propagator_tau
    # REMOVED: empty_propagator_dlr

    # Fixed zero-flux buffers
    zero_flux::QN
    conv_operand::Propagator{T}
    conv_dlr::Propagator{T}
end
```

**Changes to `GreenEvalBuffers`:**

```julia
struct GreenEvalBuffers{T <: Number, QN <: QuantumNumber}
    fwd_buffers::PropagatorEvalBuffers{T, QN}
    rev_buffers::PropagatorEvalBuffers{T, QN}

    backbone_0_to_beta_with_ann::Vector{Propagator{T}}

    contrib_tau_to_beta_reflected_pool::Dict{QN, Propagator{T}}
    reflection_dlr_pool::Dict{QN, Propagator{T}}

    ann_product_pool::Dict{QN, Propagator{T}}
    final_product_pool::Dict{QN, Propagator{T}}
    cre_product::Propagator{T}

    trace_result_buffer::Vector{Float64}

    # REMOVED: empty_propagator_tau
    # REMOVED: empty_propagator_dlr
end
```

**Update accessor functions:**

```julia
# Before: return get(buffers.result_pool, flux, buffers.empty_propagator_tau)
# After:  return buffers.result_pool[flux]  # Direct access, guaranteed to exist

function get_result(buffers::PropagatorEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.result_pool[flux]
end

function get_op_product(buffers::PropagatorEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.op_product_pool[flux]
end

# ... same for all other accessors
```

---

### Phase 6: Clean Up Operator Operations

**Files to modify:** `src/format/operator_ops.jl`

**Review and simplify empty block checks:**

The checks like `if isempty(B.blocks) return end` can be:
- Removed entirely (if we're confident flux pre-filtering catches all cases)
- Or kept as debug assertions: `@assert !isempty(B.blocks) "Unexpected empty propagator"`

**Recommendation:** Keep as `@assert` during development, remove in production.

---

### Phase 7: Update Interface Declarations

**Files to modify:** `src/format/interface.jl`

Remove declarations that are no longer needed or update signatures.

---

## Testing Strategy

### Unit Tests

1. **`check_flux_sequence` accuracy:**
   - Create diagrams with known valid/invalid sequences
   - Verify function correctly identifies each

2. **`build_valid_groups` completeness:**
   - Compare total_diagrams_in_groups with actual non-zero evaluations
   - Must match exactly

3. **Buffer access without empty sentinel:**
   - Verify all accessed fluxes are in pools
   - No KeyError exceptions

### Integration Tests

1. **Numerical correctness:**
   - Run full calculation with old and new code
   - Results must match to machine precision

2. **Performance improvement:**
   - Measure time and allocations
   - Expect 20-50% improvement depending on order

### Regression Tests

1. Existing test suite must pass unchanged
2. Add test for hierarchical iteration specifically

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/diagram/DiagramConfig.jl` | Add DiagramGroup, check_flux_sequence, build_valid_groups |
| `src/util/BatchGeneration.jl` | Add GroupBatchIterator, group_batch_iterator |
| `src/diagram/eval_propagator.jl` | Use group-based iteration, remove empty checks |
| `src/diagram/eval_green.jl` | Use group-based iteration, remove empty checks |
| `src/diagram/backbone_eval.jl` | Remove haskey early exit, remove empty_propagator returns |
| `src/format/backbone_buffers.jl` | Remove empty_propagator fields, simplify accessors |
| `src/format/interface.jl` | Update declarations |
| `src/format/operator_ops.jl` | Convert isempty checks to assertions |
| `test/check_flux_filtering.jl` | New test file for flux validation |

---

## Implementation Order

1. **Phase 1**: Add flux validation (can test independently)
2. **Phase 2**: Add group iterator (can test independently)
3. **Phase 3**: Update eval functions (integration point)
4. **Phase 4**: Simplify backbone (depends on Phase 3)
5. **Phase 5**: Remove empty propagators (depends on Phase 4)
6. **Phase 6**: Clean up operator ops (final cleanup)
7. **Phase 7**: Update interface (final cleanup)

Each phase should:
1. Implement changes
2. Run tests
3. Commit

---

## Risk Mitigation

1. **Flux validation bug could skip valid diagrams:**
   - Add assertion in backbone eval: `@assert !isempty(result.blocks)`
   - Compare results with old implementation before removing checks

2. **Group distribution could cause load imbalance:**
   - Groups have variable size (different n_exponent_combos)
   - Consider work-stealing or weighted distribution if imbalance is severe

3. **Breaking change to public API:**
   - Buffer types change (no empty_propagator)
   - Keep old accessor signatures working (return value same for valid fluxes)

---

## Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Invalid diagram evaluations | ~30-50% of total | 0% | 30-50% speedup |
| Buffer memory | +2 empty propagators | -2 propagators | Minor |
| Code complexity | Multiple validity checks | Single upfront check | Cleaner |
| Cache locality | Random access | Group-sequential | 5-15% speedup |

**Total expected improvement:** 35-65% faster for typical calculations.
