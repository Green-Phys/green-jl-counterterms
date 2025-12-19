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
- `test/bare/` - Dense format test with 1 orbital
- `test/bare_blockdense/` - BlockDense format test with 1 orbital

Both validate partition function and Green's function to 12-digit precision against Mathematica results in `test/bare/analytic.nb`.

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
julia --project=. -e 'using QMBToolkitSOEHYB; QMBToolkitSOEHYB.run("test/bare/input.toml")'
```

Note: The solver is MPI-enabled and uses `mpiexecjl` internally.

## Architecture

### Core Data Flow

1. **Input Reading** (`src/util/io.jl::read_input`) - Parses TOML into `Params` structs
2. **Model Initialization** (`src/util/init_data.jl::init_model_data` or `init_model_data_blockdense`) - Loads Hamiltonian, performs exact diagonalization, builds SOE representation
3. **Solver Initialization** (`src/util/init_data.jl::init_solver_data`) - Creates empty `CalculationData`
4. **Solve** (`src/solver/solve.jl::solve`) - Generates and evaluates diagrams, computes partition function and Green's function

### Data Structure Hierarchy

**ModelData** (`src/solver/Data.jl`):
- Hamiltonian eigenvalues/eigenvectors from exact diagonalization
- Creation/annihilation operators in energy eigenbasis
- SOE representation: `SOEData` with coefficients, exponents, and K±(τ) on DLR grid
- Bare propagator as `DLRData`
- Pre-computed caches: `KernelCache`, `ConvolutionWorkspace`
- Format-dependent fields for BlockDense operators

**CalculationData** (`src/solver/Data.jl`):
- Propagator and self-energy (format-dependent storage)
- Green's function (always dense - physical observable)
- `partition_function_by_order`, `green_by_order` arrays

**SolverData** (`src/solver/Data.jl`):
- Combines `ModelData` and `CalculationData`
- Parametrized by `{T<:Number, Format<:RepresentationFormat}`

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

### Generation and Evaluation Pipeline

1. `gen_diagrams` (`src/diagram/gen_common.jl`) - Top-level dispatcher
2. `gen_propagator_given_order` / `gen_green_given_order` - Generate topologies and flavor/SOE combinations
3. `eval_diagrams!` (`src/diagram/eval_common.jl`) - Format dispatch wrapper
4. `multiplication_convolution` (`src/dense/convolution_eval.jl` or `src/blockdense/convolution_eval.jl`) - Core evaluation

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
- Propagator order must equal Green's function order - 1

### File Organization

```
src/
├── QMBToolkitSOEHYB.jl     # Main module, run() entry point
├── solver/
│   ├── Data.jl             # ModelData, CalculationData, SolverData
│   ├── Representation.jl   # DenseFormat, BlockDenseFormat traits
│   └── solve.jl            # Main control loop
├── util/
│   ├── io.jl               # TOML parsing, HDF5 output
│   ├── Params.jl           # Parameter structs
│   ├── init_data.jl        # Model/solver initialization
│   └── others.jl           # Helper functions
├── diagram/
│   ├── DiagramStructure.jl # BackBone, BackBoneNode, BackBoneEdge
│   ├── gen_common.jl       # Common generation infrastructure
│   ├── gen_eval_propagator.jl
│   ├── gen_eval_green.jl
│   ├── gen_eval_selfenergy.jl  # Stub (not implemented)
│   ├── eval_common.jl      # Format-independent evaluation
│   └── print_diagram.jl    # Visualization
├── dense/                  # Dense matrix operations
│   ├── operator_ops.jl     # _multiplication_23!, _multiplication_32!, etc.
│   ├── hybridization.jl    # apply_hybridization_inplace!
│   ├── propagator.jl       # eval_propagator_backbone
│   └── convolution_eval.jl # multiplication_convolution
└── blockdense/             # Block-sparse operations
    ├── algebra.jl          # BlockDenseArray arithmetic
    ├── operator_ops.jl     # Block-aware multiplications
    ├── hybridization.jl    # Block-aware hybridization
    ├── propagator.jl       # Block-aware propagator evaluation
    └── convolution_eval.jl # multiplication_convolution_blockdense
```

### Representation Formats

Two formats (`src/solver/Representation.jl`):
- **DenseFormat**: Full dense matrices in Fock space (default)
- **BlockDenseFormat**: Block-sparse exploiting particle number conservation

Set via `format = "dense"` or `format = "blockdense"` in `[solver]` section.

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

[solver]
output_path = "."
format = "dense"                          # or "blockdense"
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
- Bold propagator calculation (error at `solve.jl:37`)
- Self-energy diagram evaluation (`gen_eval_selfenergy.jl` is stub)

## Dependencies

**Internal (QMBToolkit ecosystem)**:
- `QMBToolkitDLR` - DLR grid and transformations
- `QMBToolkitED` - Exact diagonalization, BlockDense types

**External**:
- `HDF5` - Output format
- `TOML` - Input parsing
- `MPI` - Parallel execution
- `LinearAlgebra`, `DelimitedFiles`