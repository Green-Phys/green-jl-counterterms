# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QMBToolkitSOEHYB is a Julia package implementing a bold hybridization expansion solver based on the Sum of Exponentials (SOE) representation. The solver computes quantum many-body propagators and Green's functions using diagrammatic expansions with Discrete Lehmann Representation (DLR) for efficient representation in imaginary time.

## Development Commands

### Environment Setup
```julia
using Pkg
Pkg.instantiate()
```

### Running Tests
```julia
using Pkg
Pkg.test()
```

The test suite runs two test cases comparing against analytical results:
- `test/data/` - Single-orbital test case
- `test/data_2spin/` - Two-spin test case with independent equivalent spins

Both validate partition function and Green's function to 12-digit precision against Mathematica results.

### Running the Solver

**Executable script:**
```bash
./bin/run_soehyb <input_file.toml> [number_of_processes]
```

**From Julia REPL:**
```julia
using QMBToolkitSOEHYB
QMBToolkitSOEHYB.run("path/to/input.toml")
```

**Direct Julia invocation (for testing):**
```bash
julia --project=. -e 'using QMBToolkitSOEHYB; QMBToolkitSOEHYB.run("test/data/input.toml")'
```

Note: The solver is MPI-enabled and uses `mpiexecjl` internally.

## Architecture

### Core Data Flow

1. **Input Reading** (`src/util/io.jl::read_input`) - Parses TOML into `Params` structs
2. **Model Initialization** (`src/util/init_data.jl::init_model_data`) - Loads Hamiltonian, performs exact diagonalization, builds SOE representation
3. **Solver Initialization** (`src/util/init_data.jl::init_solver_data`) - Creates empty `CalculationData`
4. **Solve** (`src/solver/solve.jl::solve`) - Generates and evaluates diagrams, computes partition function and Green's function

### Data Structure Hierarchy

**ModelData** (`src/solver/Data.jl`):
- Hamiltonian eigenvalues/eigenvectors from exact diagonalization
- Creation/annihilation operators in energy eigenbasis (stored in `ModelDataStorage`)
- SOE representation: `SOEData` with coefficients, exponents, and K±(τ) on DLR grid
- Bare propagator as `DLRPropagator`
- Pre-computed caches: `KernelCache`, `ConvolutionWorkspace`

**CalculationData** (`src/solver/Data.jl`):
- Propagator and self-energy as `DLRPropagator` (stored in `CalculationDataStorage`)
- Green's function as `GreenDLR` (physical observable in orbital space)
- `partition_function_by_order`, `green_numerator_by_order`, `green_by_order` arrays

**SolverData** (`src/solver/Data.jl`):
- Combines `ModelData` and `CalculationData`
- Parametrized by `{T<:Number}` (typically `ComplexF64`)

### Propagator Types

**Propagator** (`src/format/propagator.jl`):
- Block-sparse time-dependent propagator exploiting quantum number conservation
- Blocks stored as `Dict{Tuple{Int,Int}, Array{T,3}}` indexed by `(target_sector, source_sector)`
- Each block has shape `(row_dim, col_dim, time_points)` for contiguous matrix operations

**DLRPropagator** (`src/format/propagator.jl`):
- Wraps two `Propagator` instances: `vals` (values on DLR tau grid) and `coeffs` (DLR expansion coefficients)

**GreenDLR** (`src/format/GreenFunction.jl`):
- Dense Green's function in orbital space (physical observable)
- Shape: `(n_orb, n_orb, n_tau)`

### Diagram Structure

**MutableBackBone** (`src/diagram/DiagramStructure.jl`):
Diagrams are linear chains of M nodes and M-1 edges:
```
|Node(M)| --- Edge(M-1) --- |Node(M-1)| --- ... --- |Node(1)|
```

Uses zero-allocation iteration: a single `MutableBackBone` is pre-allocated and reused for all diagrams via `reset_backbone!` and `fill_backbone_from_group!`.

**BackBoneNode**:
- `opt_type`: `:c` (annihilation), `:cdag` (creation), or `:empty`
- `flavor`: orbital index
- `bar_index`: SOE index (1 to p) or -1 for undecorated
- `attached_hyb`: K⁺ₗ, K⁻ₗ, or NoHyb (encoded as `HybridizationState`)

**HybridizationState** (efficient integer encoding):
- `index == 0`: NoHyb
- `index > 0`: KPlus(index)
- `index < 0`: KMinus(-index)

### Zero-Allocation Diagram Iteration

Diagrams are generated on-the-fly with zero allocation using hierarchical grouping:

1. **Configuration Caching** (`src/diagram/DiagramConfig.jl`):
   - `build_config_cache()` - Pre-computes all topologies, directions, flavors, exponents
   - `DiagramConfigCache` - Stores configurations with dimensions `(n_topologies, n_directions, n_flavors_combo, n_exponents_combo)`
   - `build_valid_groups()` - Pre-filters diagram groups by quantum number flux compatibility
   - `DiagramGroup` - Groups diagrams sharing topology, direction, and flavor (only exponents vary)

2. **Zero-Allocation Iteration**:
   - `allocate_mutable_backbone()` - Creates single reusable backbone
   - `reset_backbone!()` / `fill_backbone_from_group!()` - Reuse without allocation
   - MPI ranks iterate over groups (not individual diagrams)

3. **Evaluation** (`src/diagram/eval_*.jl`):
   - `eval_diagrams!` - Top-level dispatcher by diagram type
   - `eval_propagator_given_order!` / `eval_green_given_order!` - Iterate groups and accumulate
   - `eval_propagator_forward_backbone` / `eval_green_forward_backbone` - Backbone evaluation
   - `multiplication_convolution` - Core convolution with fused K-factor operations

### Format Interface Pattern

The codebase uses a format-agnostic interface pattern (`src/format/interface.jl`) that allows diagram evaluation code to work with block-sparse propagators. Key interface functions:

- **Propagator Operations**: `zeros_like`, `add_scaled!`
- **DLR Operations**: `tau2dlr`, `dlr2tau`, `do_reflection_tau2tau!`, `extract_at_tau`, `do_convolution_dlr2tau!`
- **Operator Operations**: `multiply_op_prop!`, `multiply_prop_op!`, `multiply_prop_prop!`, `trace_prop!`
- **Fused Operations**: `multiply_op_prop_with_K!`, `multiply_prop_op_with_K!` (combines matrix mult with hybridization scaling)
- **Hybridization**: `apply_hybridization_inplace!`, `get_K_vector`
- **Buffer Management**: `allocate_propagator_buffers`, `allocate_green_buffers`, `get_result`, `get_op_product`, etc.
- **Flux Handling**: `get_flux`, `compute_new_flux`, `compute_new_flux_reverse`
- **MPI**: `mpi_reduce_and_add!`

Implementations are in `src/format/` files: `dlr.jl`, `operator_ops.jl`, `hybridization.jl`, `backbone_buffers.jl`.

### Key Architectural Points

**SOE Representation**: Hybridization Δᵢⱼ(τ) decomposed as:
```
Δᵢⱼ(τ) = Σₗ coeffs[l,i,j] * K(τ, exponents[l])
```
where K⁺(τ,ω) = exp(-ωτ)/(1+exp(-βω)) and K⁻(τ,-ω) for the adjoint.

**DLR Integration**:
- `DLRPropagator.vals`: Values on DLR τ-grid
- `DLRPropagator.coeffs`: DLR expansion coefficients
- Efficient τ ↔ ω transformations via QMBToolkitDLR

**Block-Sparse Structure**: Propagators exploit quantum number conservation - only blocks allowed by the flux are allocated and computed.

**Energy Shift**: Hamiltonian shifted by E_shift = min(eigenvalues) for numerical stability. Partition function corrected in post-processing.

**Bare vs Bold Expansion**:
- Both must be consistently bare or both bold
- Propagator order must equal Green's function order - 1 (propagator order and self-energy order is defined as the number of hybridization lines, while the order of Green's function is defined as the number of hybridization lines plus one.)

### File Organization

```
src/
├── QMBToolkitSOEHYB.jl       # Main module, run() entry point
├── solver/
│   ├── Data.jl               # ModelData, CalculationData, SolverData, storage types
│   └── solve.jl              # Main control loop
├── util/
│   ├── io.jl                 # TOML parsing, HDF5 output
│   ├── Params.jl             # Parameter structs
│   └── init_data.jl          # Model/solver initialization
├── diagram/
│   ├── DiagramStructure.jl   # MutableBackBone, BackBoneNode, MutableBackBoneEdge
│   ├── DiagramConfig.jl      # Configuration caching, DiagramGroup, flux pre-filtering
│   ├── gen_common.jl         # Pairing generation utilities
│   ├── eval_common.jl        # Top-level eval_diagrams! dispatcher
│   ├── eval_propagator.jl    # Propagator evaluation with zero-allocation iteration
│   ├── eval_green.jl         # Green's function evaluation with zero-allocation iteration
│   ├── backbone_eval.jl      # Unified backbone evaluation (uses interface)
│   └── print_diagram.jl      # Visualization utilities
└── format/
    ├── interface.jl          # Abstract interface declarations
    ├── GreenFunction.jl      # GreenDLR type
    ├── propagator.jl         # Propagator, DLRPropagator types
    ├── dlr.jl                # DLR transformations (tau2dlr, dlr2tau, etc.)
    ├── operator_ops.jl       # Block-aware multiplications
    ├── hybridization.jl      # Hybridization application
    └── backbone_buffers.jl   # Buffer allocation for backbone evaluation
```

### Input File Format

TOML with sections:
```toml
[model]
beta = 10.0
file_path = "."                           # Relative to input file directory
hopping_file = "hopping.txt"
interaction_file = "Uijkl.txt"
hybridization_poles_file = "hyb_poles.txt"
hybridization_weights_file = "hyb_weights.txt"

[solver]
output_path = "."

[solver.propagator]
is_bold = false
expansion_order = 2

[solver.green]
is_bold = false
expansion_order = 3                       # Must be propagator order + 1
save_linear_size = 1001                   # Output grid points

[solver.dlr]
rtol = 1e-14
Euv = 500.0
```

### Data File Formats

**hopping.txt**: `i j re im` (0-indexed orbital indices)
```
0 0 2.0 0.0
```

**Uijkl.txt**: Interaction matrix (sparse format)

**hyb_poles.txt**: One exponent (ω) per line
```
1.0
-2.0
```

**hyb_weights.txt**: `ind_poles orb_1 orb_2 weight` (0-indexed)
```
0 0 0 -3.0
1 0 0 -1.0
```
Must have complete (n_poles × n_orb × n_orb) entries.

### Output Format (HDF5)

Saved to `<output_path>/sim.h5`:
```
/green/
  linear_grid/green_tau, tau
  dlr_grid/green_tau, tau, coeffs
  by_order/green_order_1, green_order_2, ...
/partition_function/
  partition_function, partition_function_by_order, partition_function_corrected
/model/
  E_shift, beta, Z_counterterms
/timing/
  total_wall_time, total_cpu_time
```

### Checkpoint / Restart

A killed run (walltime limit, reboot, crash) resumes from completed work: rerun the
**identical command** and it continues where it stopped. Auto-resume is ON by
default with zero required config. Implementation is `src/util/checkpoint.jl`, wired
into `solve.jl` and `diagram/eval_common.jl`.

- **Checkpoint file**: `sim.ckpt.h5`, a sibling of `sim.h5`. Written atomically
  (same-dir temp + `rename`), and **deleted on successful completion** so a fresh
  rerun starts clean.
- **Fingerprint guard**: the checkpoint stores a fingerprint (beta, expansion
  orders, is_bold flags, flavor counts, DLR Euv/rtol, exponents, sector dims). On a
  mismatch the solver warns and starts fresh without deleting the stale file.
  `nprocs` is intentionally excluded — resume may use a different rank count.
- **Granularity**:
  - *Order level* (default): checkpoint after each completed expansion order.
  - *Within order*: set `[solver] checkpoint_group_interval = N` (default `0` =
    order boundaries only). Each order's diagram groups are processed in consecutive
    prefix chunks of `N` groups; after each chunk the partial sum is `Allreduce`d
    into the cumulative accumulator (every rank then identical) and the checkpoint
    records that accumulator plus the integer count of groups done. Use this when a
    single order is itself very long.
- **Reproducibility**: a resumed run is **bit-identical** to an uninterrupted run at
  the **same nprocs and same interval** (deterministic per-rank group split +
  per-block Allreduce + lossless HDF5 Float64 round-trip). Across **different
  nprocs**, or between **different intervals** (including `N>0` vs `0`), the result
  is still correct but differs at ~1e-16 relative because floating-point summation
  is regrouped — this is expected, not a bug.
- **Env vars**: `QMBTOOLKIT_NO_RESUME=1` ignores any existing checkpoint and starts
  fresh. (`QMBTOOLKIT_CRASH_PHASE` / `QMBTOOLKIT_CRASH_AFTER_ORDER` /
  `QMBTOOLKIT_CRASH_AFTER_GROUPS` are test-only hooks that simulate a mid-run kill.)
- **Scope**: covers the implemented **bare** path (propagator orders `0..P`, green
  orders `1..G`). The bold path is not implemented and not yet checkpoint-aware.
- **Tests**: `test/checkpoint_restart_test.jl` (in-process, single rank, part of
  `Pkg.test()`); `test/checkpoint_restart_mpi.jl` (manual multi-rank verification —
  run directly, launches real `mpiexec` jobs).

### Current Implementation Status

**Implemented**:
- Bare expansion for propagator (block-sparse)
- Green's function calculation (bare mode)
- DLR-based representation
- SOE decomposition of hybridization
- MPI parallelization for diagram evaluation
- Checkpoint / restart with auto-resume (order-level and within-order; see above)

**Not Yet Implemented**:
- Bold propagator calculation (error at `solve.jl:42`)
- Self-energy diagram evaluation

## Dependencies

**Internal (QMBToolkit ecosystem)**:
- `QMBToolkitDLR` - DLR grid and transformations
- `QMBToolkitED` - Exact diagonalization, Operator/Basis types

**External**:
- `HDF5` - Output format
- `TOML` - Input parsing
- `MPI` - Parallel execution
- `LinearAlgebra`, `DelimitedFiles`
