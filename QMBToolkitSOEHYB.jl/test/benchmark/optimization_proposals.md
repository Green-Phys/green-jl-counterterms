# QMBToolkitSOEHYB Optimization Proposals

## Phase 1: Hot Loop Optimizations (COMPLETED)

| ID | Target | Result |
|----|--------|--------|
| S5 | Reflected kernel cache | 25% faster, 0 allocs |
| S2 | Hybridization in-place | 55% faster |
| S1 | Convolution coefficients | Marginal |
| M1 | Valid flux computation | 42% buffer reduction |
| S7 | Workspace consolidation | Simplified code |

**End-to-end improvement:** 54.6 KiB → 23.5 KiB per diagram (~57% reduction)

---

## Phase 2: Structural Optimizations

### Implementation Status

| ID | Target | Status | Impact |
|----|--------|--------|--------|
| A4 | Share empty propagators | **DONE** | Minor memory savings |
| A2 | Remove dead code buffers | **DONE** | 2 buffer pools removed, simplified API |
| A3 | Selective flux allocation | ANALYZED | Not worth complexity |
| A1 | Unify buffer abstraction | **DONE** | Proper structs with typed accessors |

---

### A1. Unify Buffer Abstraction

**Problem:** Accessor functions in `backbone_buffers.jl:251-265` reference fields that don't exist in `PropagatorEvalBuffers` struct:
- `get_contrib_tau_to_beta_reflected()`
- `get_reflection_dlr()`
- `get_ann_product()`
- `get_final_product()`

These work via duck-typing on the NamedTuple from `allocate_green_buffers()`, but this is fragile.

**Current architecture (after A2, A4, A1):**
```
PropagatorEvalBuffers{T, QN}           GreenEvalBuffers{T, QN}
├── result_pool                        ├── fwd_buffers::PropagatorEvalBuffers{T, QN}
├── op_product_pool                    ├── rev_buffers::PropagatorEvalBuffers{T, QN}
├── result_dlr_pool                    ├── backbone_0_to_beta_with_ann::Vector{Propagator{T}}
├── conv_operand                       ├── contrib_tau_to_beta_reflected_pool::Dict{QN, Propagator{T}}
├── conv_dlr                           ├── reflection_dlr_pool::Dict{QN, Propagator{T}}
├── empty_propagator_tau               ├── ann_product_pool::Dict{QN, Propagator{T}}
└── empty_propagator_dlr               ├── final_product_pool::Dict{QN, Propagator{T}}
                                       ├── cre_product::Propagator{T}
                                       ├── trace_result_buffer::Vector{Float64}
                                       ├── empty_propagator_tau (reused from fwd_buffers)
                                       └── empty_propagator_dlr (reused from fwd_buffers)
```

**Solution options:**

**Option A: Extend PropagatorEvalBuffers for Green**
Create `GreenBuffers` that wraps PropagatorEvalBuffers and adds Green-specific pools:
```julia
struct GreenBuffers{T, QN}
    fwd::PropagatorEvalBuffers{T, QN}
    rev::PropagatorEvalBuffers{T, QN}
    contrib_tau_to_beta_reflected_pool::Dict{QN, Propagator{T}}
    reflection_dlr_pool::Dict{QN, Propagator{T}}
    ann_product_pool::Dict{QN, Propagator{T}}
    final_product_pool::Dict{QN, Propagator{T}}
    cre_product::Propagator{T}
    trace_buffer::Vector{Float64}
end
```

**Option B: Keep NamedTuple, fix accessors**
Move the Green-specific accessor functions to accept NamedTuple explicitly, document the interface.

---

### A2. Remove Dead Code Buffers - COMPLETED

**Problem found:** The `temp_buffer` parameter in `apply_hybridization_inplace!` was never used - the function modified `target` directly. This meant:
- `hyb_temp_pool` in PropagatorEvalBuffers was allocated but never used
- `conv_hyb_temp` buffer was allocated but never used

**Solution implemented:**

1. Simplified `apply_hybridization_inplace!` signature:
   ```julia
   # Before: apply_hybridization_inplace!(temp_buffer, target, hyb, model_data; is_reverse)
   # After:  apply_hybridization_inplace!(target, hyb, model_data; is_reverse)
   ```

2. Removed from `PropagatorEvalBuffers`:
   - `hyb_temp_pool` (entire Dict of buffers)
   - `conv_hyb_temp` (single buffer)

3. Updated all call sites in `backbone_eval.jl`

**Result:** Reduced from 7 buffer roles to 5, simplified API, reduced memory usage.

---

### A3. Selective Flux Allocation for Green Buffers - ANALYZED (Not Implemented)

**Problem:** `allocate_green_buffers()` allocates pools for ALL valid fluxes.

**Analysis performed:**

1. **`final_product_pool`** - Already selective (only allocates for unique `ann_opt.flux` values)

2. **`ann_product_pool`** - The flux depends on `contrib_0_to_tau`'s flux, which varies per diagram. We can't know in advance which diagrams will produce non-zero contributions.

3. **`contrib_tau_to_beta_reflected_pool`** and **`reflection_dlr_pool`** - Same issue: flux depends on diagram evaluation results.

**Correctness concern:** If we don't allocate for a flux that IS valid and IS needed, the accessor returns `empty_propagator_tau`. Subsequent operations (`fill!`, `multiply_op_prop!`) silently produce zero, leading to incorrect results.

**Conclusion:** The current allocation strategy (all valid fluxes) is necessary for correctness. Selective allocation would require either:
- Complex static analysis of all possible diagram structures
- Runtime lazy allocation with Dict resize overhead

Neither option provides enough benefit to justify the complexity. The current approach is safe and reasonably efficient.

---

### A4. Share Empty Propagators

**Problem:** Empty propagators are allocated twice:
1. In `PropagatorEvalBuffers` constructor (lines 175-176)
2. In `allocate_green_buffers()` (lines 329-330)

**Solution:** Create once and share:
```julia
# In allocate_green_buffers():
fwd_buffers = PropagatorEvalBuffers{T}(basis, max_order, n_tau, n_dlr)
rev_buffers = PropagatorEvalBuffers{T}(basis, max_order, n_tau, n_dlr)

# Reuse from fwd_buffers instead of creating new ones
empty_propagator_tau = fwd_buffers.empty_propagator_tau
empty_propagator_dlr = fwd_buffers.empty_propagator_dlr
```

---

## Implementation Summary

| ID | Status | Notes |
|----|--------|-------|
| A4 | **DONE** | Reuse empty propagators from fwd_buffers |
| A2 | **DONE** | Removed 2 unused buffer pools, simplified hybridization API |
| A3 | **ANALYZED** | Not worth complexity - current allocation is necessary for correctness |
| A1 | **DONE** | Created GreenEvalBuffers struct with typed accessors |

---

## Files Modified (Phase 2)

| File | Changes |
|------|---------|
| `src/format/backbone_buffers.jl` | A4: Reuse empty propagators; A2: Remove hyb_temp_pool, conv_hyb_temp |
| `src/format/hybridization.jl` | A2: Simplified apply_hybridization_inplace! signature |
| `src/format/interface.jl` | A2: Removed unused function declarations |
| `src/diagram/backbone_eval.jl` | A2: Updated hybridization call sites |
| `test/benchmark/benchmark_core.jl` | A2: Updated benchmark calls |
