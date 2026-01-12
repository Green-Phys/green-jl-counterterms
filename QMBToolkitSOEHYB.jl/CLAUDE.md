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
- `test/bare_dense/` - Dense format test with 1 orbital
- `test/bare_blockdense/` - BlockDense format test with 1 orbital

Both validate partition function and Green's function to 12-digit precision against Mathematica results in `test/bare_dense/analytic.nb`.

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
julia --project=. -e 'using QMBToolkitSOEHYB; QMBToolkitSOEHYB.run("test/bare_dense/input.toml")'
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
- Creation/annihilation operators in energy eigenbasis
- SOE representation: `SOEData` with coefficients, exponents, and K±(τ) on DLR grid
- Bare propagator as `DLRData`
- Pre-computed caches: `KernelCache`, `ConvolutionWorkspace`
- Format-dependent fields for operators

**CalculationData** (`src/solver/Data.jl`):
- Propagator and self-energy (format-dependent storage)
- Green's function (always dense - physical observable)
- `partition_function_by_order`, `green_by_order` arrays

**SolverData** (`src/solver/Data.jl`):
- Combines `ModelData` and `CalculationData`
- Parametrized by `{T<:Number, Format<:HilbertSpaceFormat}`

### Diagram Structure

**BackBone** (`src/diagram/DiagramStructure.jl`):
Diagrams are linear chains of M nodes and M-1 edges:
```
|Node(M)| --- Edge(M-1) --- |Node(M-1)| --- ... --- |Node(1)|
```

**BackBoneNode**:
- `opt_type`: `:c` (annihilation), `:cdag` (creation), or `:empty`
- `flavor`: orbital index
- `bar_index`: SOE index (1 to p) or -1 for undecorated
- `attached_hyb`: K⁺ₗ, K⁻ₗ, or NoHyb

**HybridizationState** (efficient integer encoding):
- `index == 0`: NoHyb
- `index > 0`: KPlus(index)
- `index < 0`: KMinus(-index)

### Batched Generation and Evaluation Pipeline

Diagrams are generated and evaluated in batches for memory efficiency and MPI distribution:

1. **Configuration Caching** (`src/diagram/DiagramConfig.jl`):
   - `build_config_cache()` - Pre-computes all topologies, directions, flavors, exponents
   - `DiagramConfigCache` - Stores configurations with dimensions `(n_topologies, n_directions, n_flavors_combo, n_exponents_combo)`
   - `generate_diagram_at_index()` - Generates a single diagram from linear index

2. **Batch Iteration** (`src/util/BatchGeneration.jl`):
   - `BatchIterator` - Iterates through diagrams in batches with MPI distribution
   - `batch_iterator()` - Creates iterator for a specific MPI rank's portion

3. **Evaluation** (`src/diagram/eval_*.jl`):
   - `eval_diagrams!` - Top-level dispatcher by diagram type
   - `eval_propagator_given_order!` / `eval_green_given_order!` - Iterate batches and accumulate
   - `eval_backbone` (`src/diagram/backbone_eval.jl`) - Unified backbone evaluation using interface
   - `multiplication_convolution` (`src/format/*/propagator.jl`) - Format-specific core evaluation

### Format Interface Pattern

The codebase uses a format-agnostic interface pattern (`src/format/interface.jl`) that allows diagram evaluation code to work with both Dense and BlockDense formats. Key interface functions:

- **Propagator Operations**: `zeros_like`, `add_scaled!`, `get_data`
- **DLR Operations**: `tau2dlr`, `dlr2tau`, `do_reflection_tau2tau`, `extract_at_tau`, `do_convolution_dlr2tau!`
- **Operator Operations**: `multiply_op_prop!`, `multiply_prop_op!`, `multiply_prop_prop!`, `trace_prop!`
- **Convolution**: `multiplication_convolution`, `multiplication_convolution_reverse`
- **Buffer Management**: `allocate_propagator_buffers`, `allocate_green_buffers`, `get_result`, `get_hyb_temp`, etc.
- **MPI**: `mpi_reduce_and_add!`

Each format (Dense/BlockDense) implements these interfaces in its respective subdirectory under `src/format/`.

### Key Architectural Points

**SOE Representation**: Hybridization Δᵢⱼ(τ) decomposed as:
```
Δᵢⱼ(τ) = Σₗ coeffs[l,i,j] * K(τ, exponents[l])
```
where K⁺(τ,ω) = exp(-ωτ)/(1+exp(-βω)) and K⁻(τ,-ω) for the adjoint.

**DLR Integration**:
- `DLRData.vals`: Values on DLR τ-grid
- `DLRData.coeffs`: DLR expansion coefficients
- Efficient τ ↔ ω transformations via QMBToolkitDLR

**Energy Shift**: Hamiltonian shifted by E_shift = min(eigenvalues) for numerical stability. Partition function corrected in post-processing.

**Bare vs Bold Expansion**:
- Both must be consistently bare or both bold
- Propagator order must equal Green's function order - 1 (propagator order and self-energy order is defined as the number of hybridization lines, while the order of Green's function is defined as the number of hybridziation liens plus one.)

### File Organization

```
src/
├── QMBToolkitSOEHYB.jl       # Main module, run() entry point
├── solver/
│   ├── Data.jl               # ModelData, CalculationData, SolverData
│   ├── HilbertSpaceFormat.jl # DenseFormat, BlockDenseFormat type traits
│   └── solve.jl              # Main control loop
├── util/
│   ├── io.jl                 # TOML parsing, HDF5 output
│   ├── Params.jl             # Parameter structs
│   ├── init_data.jl          # Model/solver initialization
│   └── BatchGeneration.jl    # Batched diagram generation with MPI distribution
├── diagram/
│   ├── DiagramStructure.jl   # BackBone, BackBoneNode, BackBoneEdge
│   ├── DiagramConfig.jl      # Configuration caching for batched generation
│   ├── gen_common.jl         # Pairing generation utilities
│   ├── eval_common.jl        # Top-level eval_diagrams! dispatcher
│   ├── eval_propagator.jl    # Propagator evaluation with batching
│   ├── eval_green.jl         # Green's function evaluation with batching
│   ├── backbone_eval.jl      # Unified backbone evaluation (uses interface)
│   ├── print_diagram.jl      # Visualization utilities
│   ├── eval_selfenergy.jl    # Stub (not implemented)
│   └── eval_dyson.jl         # Stub (not implemented)
└── format/                   # Format-specific implementations
    ├── interface.jl          # Abstract interface for format-agnostic operations
    ├── dense/                # Dense matrix operations
    │   ├── DensePropagator.jl    # DensePropagator type
    │   ├── dlr.jl                # DLR transformations (tau2dlr, dlr2tau, etc.)
    │   ├── operator_ops.jl       # multiply_op_prop!, multiply_prop_op!, etc.
    │   ├── hybridization.jl      # apply_hybridization_inplace!
    │   └── propagator.jl         # multiplication_convolution, buffer allocation
    └── blockdense/           # Block-sparse operations
        ├── BlockDensePropagator.jl  # BlockDensePropagator type
        ├── dlr.jl                   # Block-aware DLR transformations
        ├── operator_ops.jl          # Block-aware multiplications
        ├── hybridization.jl         # Block-aware hybridization
        └── propagator.jl            # Block-aware convolution, buffer allocation
```

### Representation Formats

Two formats (`src/solver/HilbertSpaceFormat.jl`):
- **DenseFormat**: Full dense matrices in Fock space (default)
- **BlockDenseFormat**: Block-sparse exploiting particle number conservation

Set via `Hilbert_space_format = "dense"` or `Hilbert_space_format = "blockdense"` in `[solver]` section.

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
Hilbert_space_format = "dense"            # or "blockdense"
batch_size = 100000                       # Diagrams per batch (default: 100000)

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

### Current Implementation Status

**Implemented**:
- Bare expansion for propagator (dense and blockdense)
- Green's function calculation (bare mode, both formats)
- DLR-based representation
- SOE decomposition of hybridization
- MPI parallelization for diagram evaluation

**Not Yet Implemented**:
- Bold propagator calculation (error at `solve.jl:42`)
- Self-energy diagram evaluation (`eval_selfenergy.jl` and `eval_dyson.jl` are stubs)

## Dependencies

**Internal (QMBToolkit ecosystem)**:
- `QMBToolkitDLR` - DLR grid and transformations
- `QMBToolkitED` - Exact diagonalization, BlockDense types

**External**:
- `HDF5` - Output format
- `TOML` - Input parsing
- `MPI` - Parallel execution
- `LinearAlgebra`, `DelimitedFiles`