# Optimization Proposals (Post-Benchmark Analysis)

These proposals are based on benchmark data collected from the 3 orbital × 2 spin test case
with 3 SOE poles, Euv=5, rtol=0.01 (16 DLR points), propagator order 2, Green order 3.

## Implementation Status

| ID | Target | Status | Speed Impact | Alloc Reduction |
|----|--------|--------|--------------|-----------------|
| S5 | Reflected kernel cache | **DONE** | 25% on reflection | 36.5 KiB → 0 per call |
| S2 | Hybridization in-place | **DONE** | 55% on hybridization | - |
| S1 | Convolution coefficients | **DONE** | Marginal | - |
| M1 | Valid flux computation | **DONE** | Early exit for zero diagrams | ~42% buffer reduction |
| S7 | Reshape elimination | OPEN | ~1% | ~456 bytes/call |

---

## Completed Optimizations

### S5. Cache reflected kernel matrix - COMPLETED
**Commits:** `de693fb` (Lehmann.jl), `5445660` + `899815b` (QMBToolkitDLR.jl), `e994b88` (QMBToolkitSOEHYB.jl)

Pre-computed `kernel_tau_reflected = K(β - τ, ω)` once in `ConvolutionWorkspace` and added
`dlr2tau!(dest, coeffs, precomputed_kernel, workspace)` variant to Lehmann.jl.

**Results:**
- Reflection: 133 μs → 101 μs (25% faster)
- Allocations: 36.5 KiB → 0 KiB per call (100% reduction)

### S2. Hybridization in-place multiplication - COMPLETED
**Commit:** `c406af9` (QMBToolkitSOEHYB.jl)

Replaced copy-to-temp-then-copy-back with direct in-place `target_block[:, :, k] .*= vec[k]`.

**Results:**
- Hybridization: 4.33 μs → 1.96 μs (55% faster)

### S1. Pre-compute convolution coefficient matrix - COMPLETED
**Commit:** `899815b` (QMBToolkitDLR.jl)

Added `convolution_coeffs[k,j] = values[k] / (ω[k] - ω[j])` matrix to `KernelCache`,
eliminating per-iteration division in convolution inner loop.

**Results:**
- Marginal speedup (division overhead was small)

### M1. Compute valid flux set from sector structure - COMPLETED

**Files modified:**
- `QMBToolkitSOEHYB.jl/src/format/backbone_buffers.jl`
- `QMBToolkitSOEHYB.jl/src/format/operator_ops.jl`
- `QMBToolkitSOEHYB.jl/src/diagram/backbone_eval.jl`
- `QMBToolkitSOEHYB.jl/src/diagram/eval_propagator.jl`
- `QMBToolkitSOEHYB.jl/src/diagram/eval_green.jl`

**Original problem:** `enumerate_possible_flux(QN_NSz, max_order)` generates 85 flux values
for order 3, but only 49 have non-empty blocks. The 36 empty flux values waste memory.

**Initial blocker:** Intermediate fluxes during diagram evaluation can exceed what the sector
structure supports. For example, processing n annihilation operators first accumulates flux
`QN_N(n=-n)`, which may not correspond to any valid (target, source) sector pair.

**Solution implemented:** Combined approach with early exit and type stability:

1. **Use `compute_valid_fluxes(basis)`** for buffer allocation - only allocates buffers for
   fluxes that have at least one valid (target, source) sector pair.

2. **Early exit on invalid flux** - In `backbone_eval.jl`, check if the computed flux exists
   in the buffer pool before proceeding:
   ```julia
   if !haskey(buffers.op_product_pool, new_flux)
       return buffers.empty_propagator_tau  # Type-stable early exit
   end
   ```

3. **Relaxed flux assertions** - In `operator_ops.jl`, allow flux mismatch for empty propagators:
   ```julia
   @assert isempty(temp_buffer.blocks) || temp_buffer.flux == result_flux
   ```

4. **Skip empty contributions** - In evaluation functions, check for empty results:
   ```julia
   if isempty(contribution.blocks)
       continue  # Skip accumulation - diagram contributes zero
   end
   ```

**Key insight:** When intermediate flux exceeds the Hilbert space capacity (|ΔN| > x where x
is the number of spin-orbitals), no valid sector pair exists, so the propagator has no blocks.
Operations on empty propagators produce zero, and the diagram contributes nothing. By detecting
this early, we skip unnecessary computation.

**Results:**
- ~42% reduction in buffer allocations (from 85 to 49 fluxes at order 3)
- Early exit skips remaining computation for zero-contribution diagrams
- Type-stable implementation (always returns `Propagator{T}`)

---

## Open Optimizations

### S7. Eliminate ReshapedArray allocations in convolution view path
**File:** `QMBToolkitDLR.jl/src/convolution.jl`
**Status:** OPEN - Low priority after profiling

**Problem:** The `_do_convolution_view!` function creates ReshapedArray objects via
`reshape(@view(...))` calls (lines 204-208). Measured allocation: **~464 bytes per call**.

**Current assessment:**
After implementing S5, the per-call allocations dropped significantly. The reshape overhead
is now ~464 bytes per convolution call, which is much smaller than originally estimated.
With 64 allocations totaling 6.0 KiB per propagator convolution, the reshape wrappers
account for a small portion.

**Possible approaches:**
1. Cache reshaped views for each unique dimension combination
2. Use `unsafe_wrap` to create array views without allocation
3. Restructure to operate directly on linear buffers

**Priority:** Low - The absolute allocation is small and not a bottleneck.

---

## Benchmark Summary (Post-Optimization)

**Micro-benchmarks (per-call measurements):**

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| `apply_hybridization_inplace!` | 4.33 μs | 1.96 μs | 55% faster |
| `do_reflection_tau2tau!` (cached) | 133 μs, 36.5 KiB | 101 μs, 0 KiB | 25% faster, 0 allocs |
| `do_convolution_dlr2tau!` | 283 μs, 6.0 KiB | 286 μs, 6.0 KiB | No change |

**End-to-end (Green order 3, 19,440 diagrams):**

| Metric | Before | After |
|--------|--------|-------|
| Time per diagram | 1.89 ms | 1.86 ms |
| Allocations per diagram | 54.6 KiB | 23.5 KiB |
| Total allocations | ~1,036 MiB | ~446 MiB |

---

## Discussion Points

### Why S7 has lower priority now

After S5 eliminated the 36.5 KiB reflection allocations, the remaining per-call allocations
in convolution (~464 bytes for reshape wrappers) are much smaller than originally estimated.
The 6.0 KiB measured includes Dict iteration overhead, not just reshape.

**Question for discussion:**
1. Is further allocation reduction worth the added complexity?
2. Should we focus on other optimization opportunities instead?
