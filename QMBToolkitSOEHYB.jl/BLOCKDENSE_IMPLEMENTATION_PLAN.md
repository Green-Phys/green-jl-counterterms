# BlockDense Implementation Plan

**Date**: 2025-12-11
**Goal**: Add BlockDense support to QMBToolkitSOEXCA via parametric type system

## Overall Architecture

```
QMBToolkitED/          (Minimal changes - just provides operators)
QMBToolkitDLR/
  src/
    dense/             (Current DLR operations, refactored)
    blockdense/        (New BlockDense DLR operations)
QMBToolkitSOEXCA/
  src/
    dense/             (Current diagram operations, refactored)
    blockdense/        (New BlockDense diagram operations)
```

## Key Design Decisions

1. **No ED Library Changes**: Propagator construction moved to SOEXCA
2. **Folder Organization**: dense/ and blockdense/ folders in DLR and SOEXCA
3. **Safe Incremental Approach**: Each phase independently testable
4. **Format Traits**: Use `DenseFormat` and `BlockDenseFormat` for compile-time dispatch

---

## Phase 1: Infrastructure (100% Safe - No Behavior Change)

**Goal**: Add folders, traits, and move existing code into organized structure. All tests pass.

### 1.1 Add Representation Traits to SOEXCA

**New file**: `src/solver/Representation.jl`
- Define `RepresentationFormat` abstract type
- Define `DenseFormat` and `BlockDenseFormat` concrete types
- Add trait dispatch helpers: `representation_format(::Val{:dense})` etc.

**Update**: `src/QMBToolkitSOEXCA.jl`
- Include `Representation.jl` before `Data.jl`

**Testing**: `julia --project=. -e 'using Pkg; Pkg.test()'`

### 1.2 Create Dense Folder in DLR Library

**Folder structure**:
```
QMBToolkitDLR/src/
  workspace.jl          (rename from itopts.jl - only ConvolutionWorkspace, KernelCache)
  dense/
    tau2dlr.jl
    dlr2tau.jl
    convolution.jl
    reflection.jl
  blockdense/
    stubs.jl            (error stubs)
```

**Files to create**:
- Extract tau2dlr operations to `dense/tau2dlr.jl`
- Extract dlr2tau operations to `dense/dlr2tau.jl`
- Extract convolution operations to `dense/convolution.jl`
- Extract reflection operations to `dense/reflection.jl`
- Create `blockdense/stubs.jl` with error messages

**Update**: `src/QMBToolkitDLR.jl`
- Include all new files
- Export BlockDense stub functions

**Testing**: DLR tests in isolation

### 1.3 Create Dense Folder in SOEXCA Library

**Folder structure**:
```
QMBToolkitSOEXCA/src/
  dense/
    propagator.jl        (build_bare_propagator_dense, compute_bare_propagator_error)
    operator_ops.jl      (_multiplication_23!, etc.)
    convolution_eval.jl  (multiplication_convolution, multiplication_convolution_reverse)
    hybridization.jl     (apply_hybridization_inplace_dense!)
  blockdense/
    stubs.jl             (error stubs)
```

**Files to create**:
- Extract propagator construction to `dense/propagator.jl`
- Extract matrix multiplications to `dense/operator_ops.jl`
- Extract convolution evaluation to `dense/convolution_eval.jl`
- Extract hybridization to `dense/hybridization.jl`
- Create `blockdense/stubs.jl` with error messages

**Update**: `src/QMBToolkitSOEXCA.jl`
- Include all new files

**Testing**: Full SOEXCA test suite

---

## Phase 2: Parametric Data Structures (Safe - Dense Only)

**Goal**: Make data structures generic over Format, but only instantiate with DenseFormat.

### 2.1 Add Type Aliases

**New file**: `src/solver/OperatorTypes.jl`
- Define `OperatorVector{T, Format}`
- Define `SingleOperator{T, Format}`
- Define `PropagatorStorage{T, Format}`

### 2.2 Update Data Structures

**Update**: `src/solver/Data.jl`
- `DLRData{T, Format}` with format-dependent storage
- `ModelData{T, Format}` with format-dependent operators and propagators
- `CalculationData{T, Format}` with format-dependent results
- `SolverData{T, Format}` combining both

**Key**: Green's function always dense (physical observable)

### 2.3 Update Initialization

**Update**: `src/util/init_data.jl`
- Add format parameter check (error if blockdense)
- Use `DenseFormat` for all construction
- Return `ModelData{T, DenseFormat}(...)`

**Testing**: Full test suite (using DenseFormat only)

---

## Phase 3: Dispatch Infrastructure (Safe - Dense Only)

**Goal**: Make eval_common.jl dispatch to dense/ implementations. Still only Dense works.

### 3.1 Update eval_common.jl

**Update**: `src/diagram/eval_common.jl`
- Add dispatch wrappers that check Format parameter
- Call dense/ implementations for DenseFormat
- Error for BlockDenseFormat
- Keep legacy function signatures for backward compatibility

**Pattern**:
```julia
function operation!(args..., model_data::ModelData{T, Format}) where {T, Format}
    if Format == DenseFormat
        return operation_dense!(args...)
    else
        error("BlockDense not yet implemented")
    end
end
```

**Testing**: All tests pass

---

## Phase 4: Implement BlockDense in DLR Library

**Goal**: Add full BlockDense support to DLR library. Test in isolation.

### 4.1 Implement BlockDense DLR Operations

**New files**:
- `blockdense/tau2dlr.jl`: Transform each block independently
- `blockdense/dlr2tau.jl`: Inverse transformation
- `blockdense/convolution.jl`: Block-aware convolution
- `blockdense/reflection.jl`: Reflection operation

**Key insight**: Block structure preserved through DLR transforms

**Update**: `src/QMBToolkitDLR.jl`
- Replace stubs with real implementations

**Testing**: Isolated DLR BlockDense tests
- Create `test/test_dlr_blockdense.jl`
- Test roundtrip accuracy
- Test convolution correctness

---

## Phase 5: Implement BlockDense in SOEXCA

**Goal**: Enable full BlockDense support in SOEXCA.

### 5.1 Implement BlockDense Propagator

**New file**: `src/blockdense/propagator.jl`
- `build_bare_propagator_blockdense`: Construct from BlockEigenResult
- Uses block-diagonal structure (each sector independent)

### 5.2 Implement BlockDense Operations

**New file**: `src/blockdense/operator_ops.jl`
- `multiplication_23_blockdense!`: Use native BlockDense algebra
- `multiplication_32_blockdense!`: Similarly
- `trace_3_blockdense!`: Sum traces of diagonal blocks

### 5.3 Implement BlockDense Convolution Eval

**New file**: `src/blockdense/convolution_eval.jl`
- `multiplication_convolution_blockdense`: Use DLR library functions
- `multiplication_convolution_reverse_blockdense`: Similarly

### 5.4 Implement BlockDense Hybridization

**New file**: `src/blockdense/hybridization.jl`
- `apply_hybridization_inplace_blockdense!`: Apply SOE hybridization

### 5.5 Update Dispatch

**Update**: `src/diagram/eval_common.jl`
- Add `elseif Format == BlockDenseFormat` branches
- Call blockdense/ implementations

### 5.6 Update Initialization

**Update**: `src/util/init_data.jl`
- Add `init_model_data_blockdense` function
- Build Hamiltonian using `:blockdense` method
- Construct propagator using `build_bare_propagator_blockdense`
- SOE operators remain dense (convert from BlockDense if needed)

**Testing**: End-to-end BlockDense tests
- Same system, both formats
- Compare results (should match within tolerance)
- Benchmark performance

---

## Testing Strategy

### Phase-by-Phase Safety

| Phase | What Changes | Risk | Testing |
|-------|-------------|------|---------|
| **1** | Folders, traits, stubs | **Zero** | All existing tests pass |
| **2** | Parametric types (Dense only) | **Very Low** | All existing tests pass |
| **3** | Dispatch (Dense only) | **Low** | All existing tests pass |
| **4** | BlockDense DLR (isolated) | **Medium** | New DLR BlockDense tests |
| **5** | BlockDense SOEXCA | **Medium-High** | New SOEXCA BlockDense tests + comparison |

### Test Checkpoints

Each phase has explicit testing checkpoints:
- Phase 1: After each file organization step
- Phase 2: After each data structure update
- Phase 3: After dispatch refactoring
- Phase 4: After each DLR BlockDense operation
- Phase 5: After full BlockDense integration

### Rollback Strategy

- Each phase is a separate commit
- Tests must pass before moving to next phase
- Clear error messages when BlockDense not ready
- Can revert to any phase if issues found

---

## Implementation Notes

### SOE Operators

SOE operators (`ann_opts_SOE`, `cre_opts_SOE`) remain **always dense** because:
1. They operate in coefficient space (small, p × n_flavors)
2. Hybridization attachment is a small coefficient operation
3. Converting for this step is cheaper than keeping block-sparse

### DLR Transformations

BlockDense DLR transformations work by:
1. Transforming each block independently
2. Block structure preserved: if block (i→j) exists in tau-space, it exists in DLR-space
3. Avoids expensive full dense conversion

### Convolution

BlockDense convolution is efficient because:
1. Only compute blocks with compatible quantum numbers
2. Block triple (i→j→k) must satisfy quantum number conservation
3. Many potential blocks are zero by symmetry

### Green's Function

Green's function is **always dense** because:
1. It's a physical observable (small orbital space)
2. No efficiency gain from block-sparse in small matrices
3. Simplifies output and post-processing

---

## Current Status

**Phase**: Not started
**Last checkpoint**: N/A
**Next step**: Begin Phase 1.1 (Add Representation Traits)

---

## References

- SOEXCA: `/Users/yangyu/src/QMBToolkit/QMBToolkitSOEXCA.jl`
- DLR: `/Users/yangyu/src/QMBToolkit/QMBToolkitDLR.jl`
- ED: `/Users/yangyu/src/QMBToolkit/QMBToolkitED.jl`

## Usage After Implementation

```toml
# In input.toml
[model]
beta = 10.0
# ... other parameters ...
representation_format = "blockdense"  # or "dense" (default)
```

```julia
# Dense format (current)
julia> run("examples/non_interacting_Bethe/input.toml")

# BlockDense format (after implementation)
julia> run("examples/non_interacting_Bethe/input_blockdense.toml")
```

Expected performance improvement: 10-100x for large systems (n_flavors ≥ 6)
