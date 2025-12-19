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

### Running the Solver
The main entry point is the `run_soexca` executable:
```bash
./bin/run_soexca <input_file.toml> [number_of_processes]
```

Examples:
```bash
# Single process (serial execution)
./bin/run_soexca examples/spinless_semicircular/input.toml

# Multiple processes (parallel execution via MPI)
./bin/run_soexca examples/spinless_semicircular/input.toml 4
```

Note: The solver is MPI-enabled and uses `mpiexecjl` internally. Make sure MPI.jl is properly configured.

### Loading the Package in REPL
```julia
using QMBToolkitSOEHYB
QMBToolkitSOEHYB.run("path/to/input.toml")
```

## Architecture

### Core Data Flow

The solver follows a sequential pipeline:

1. **Input Reading** (`src/util/io.jl::read_input`) - Parses TOML configuration into `Params` structs
2. **Model Initialization** (`src/util/init_data.jl::init_model_data`) - Loads model data from files and constructs SOE representation
3. **Solver Initialization** (`src/util/init_data.jl::init_solver_data`) - Initializes calculation data structures
4. **Solve** (`src/solver/solve.jl::solve`) - Main calculation loop

### Data Structure Hierarchy

**ModelData** (`src/solver/Data.jl`):
- Contains model information: Hamiltonian operators, hybridization function, DLR grid
- Stores SOE representation: `SOEData` with coefficients, exponents, and K±(τ) values
- Holds exact diagonalization results: eigenvalues, eigenvectors, creation/annihilation operators

**CalculationData** (`src/solver/Data.jl`):
- Mutable container for results: propagator, self-energy, Green's function
- All quantities stored as `DLRData` (values on DLR τ-grid + DLR coefficients)

**SolverData** (`src/solver/Data.jl`):
- Combines `ModelData` and `CalculationData`
- Main data structure passed through the solver pipeline

### Diagram Generation and Evaluation

The package implements a sophisticated diagrammatic expansion system:

**BackBone Structure** (`src/diagram/DiagramStructure.jl`):
- Diagrams are represented as linear chains (backbones) of nodes and edges
- Each diagram consists of M nodes connected by M-1 edges
- Nodes contain creation/annihilation operators and attached hybridization functions
- Edges contain hybridization lines connecting operators
- Three diagram types: `:propagator_diagram`, `:selfenergy_diagram`, `:green_diagram`

**Hybridization States**:
- `NoHyb()` - No hybridization attached (encoded as index=0)
- `KPlus(l)` - Decomposed hybridization K⁺ₗ(τ) (encoded as index=l, l>0)
- `KMinus(l)` - Decomposed hybridization K⁻ₗ(τ) (encoded as index=-l, l>0)

Note: Uses efficient integer encoding instead of abstract types for optimal performance.

**Generation Pipeline**:
1. `gen_diagrams` (`src/diagram/gen_common.jl`) - Top-level: generates all diagrams for given order
2. `gen_propagator_given_order` or `gen_green_given_order` - Order-specific generation
3. `gen_diagram_combinations` - Generates all valid hybridization attachments
4. `eval_diagrams!` - Evaluates all diagrams and accumulates into `CalculationData`

### Key Architectural Points

**SOE Representation**: The hybridization function Δ(τ) is decomposed as:
```
Δᵢⱼ(τ) = Σₗ coeffs[l,i,j] * K(τ, exponents[l])
```
where K(τ,ω) = exp(-ω*τ)/(1+exp(-β*ω)) for K⁺ and K(τ,-ω) for K⁻.

**Bare vs Bold Expansion**:
- Bare expansion: Expands using unperturbed propagator
- Bold expansion: Expands using self-consistently determined propagator
- Both propagator and Green's function must use same expansion type

**DLR Integration**: All imaginary-time quantities are represented using DLR:
- `DLRData.vals`: Values on DLR τ-grid (for efficient evaluation)
- `DLRData.coeffs`: DLR expansion coefficients (for transformations)

### File Organization

- `src/QMBToolkitSOEHYB.jl` - Main module, defines `run()` entry point
- `src/util/` - Utilities: I/O, parameter handling, data initialization
- `src/solver/` - Core solver logic and data structures
- `src/diagram/` - Diagram generation and evaluation
  - `DiagramStructure.jl` - Data types for diagram representation
  - `gen_*.jl` - Diagram generation algorithms
  - `eval_*.jl` - Diagram evaluation algorithms
  - `print_diagram.jl` - Diagram visualization utilities
- `examples/` - Example calculations with input files
- `bin/run_soexca` - Executable script for running calculations

### Input File Format

TOML format with sections:
- `[model]` - Physical parameters: β, file paths for Hamiltonian, hybridization, etc.
- `[solver.propagator]` - Propagator expansion: order, bare/bold
- `[solver.green]` - Green's function expansion: order, bare/bold
- `[solver.dlr]` - DLR parameters: Euv (UV cutoff), rtol (tolerance)
- `[solver]` - Output path

See `examples/non_interacting_Bethe/input.toml` for reference.

### Current Implementation Status

**Implemented**:
- Bare expansion for propagator
- Green's function calculation (bare and bold modes)
- DLR-based representation
- SOE decomposition of hybridization

**Not Yet Implemented** (will error):
- Bold propagator calculation (raises error in `solve.jl:18`)
- Self-energy diagram evaluation (`gen_eval_selfenergy.jl` is stub)

## Dependencies

Key external packages:
- `QMBToolkitDLR` - Discrete Lehmann Representation utilities
- `QMBToolkitED` - Exact diagonalization solver
- `HDF5` - Output file format
- `QuadGK` - Numerical integration