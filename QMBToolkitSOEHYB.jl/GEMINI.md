# QMBToolkitSOEHYB

This is a Julia project that implements a bold hybridization expansion solver based on the Sum of Exponentials (SOE) representation. The solver computes quantum many-body propagators and Green's functions using diagrammatic expansions with Discrete Lehmann Representation (DLR) for efficient representation in imaginary time.

## Project Overview

*   **Goal**: Solve quantum many-body impurity problems using hybridization expansion.
*   **Method**: Sum of Exponentials (SOE) for hybridization decomposition, Discrete Lehmann Representation (DLR) for imaginary time.
*   **Implementation**: Julia, with MPI support for parallel diagram evaluation.
*   **Status**:
    *   **Implemented**: Bare expansion for propagator and Green's function (Dense and BlockDense formats), SOE decomposition, DLR integration.
    *   **Pending**: Bold expansion for propagator and self-energy diagram evaluation.

## Building and Running

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
The main entry point is the `run_soehyb` executable script, which handles MPI execution.

**Usage:**
```bash
./bin/run_soehyb <input_file.toml> [number_of_processes]
```

**Examples:**
```bash
# Serial execution
./bin/run_soehyb examples/spinless_two_exponenetials/input.toml

# Parallel execution (4 processes)
./bin/run_soehyb examples/spinless_two_exponenetials/input.toml 4
```

### REPL Usage
```julia
using QMBToolkitSOEHYB
QMBToolkitSOEHYB.run("path/to/input.toml")
```

## Architecture

### Core Data Flow
1.  **Input Reading** (`src/util/io.jl`): Parses TOML configuration into `Params`.
2.  **Initialization** (`src/util/init_data.jl`):
    *   `init_model_data`: Loads Hamiltonian, hybridization, and constructs SOE representation.
    *   `init_solver_data`: Prepares data structures for calculation results.
3.  **Diagram Generation** (`src/diagram/gen_common.jl`): Generates valid diagrams (backbones) based on expansion order and diagram type.
4.  **Diagram Evaluation** (`src/diagram/eval_common.jl`): Evaluates diagrams and accumulates results into `CalculationData` using MPI.
5.  **Data Saving**: Results are saved to HDF5 format.

### Key Data Structures

*   **ModelData**: Stores static model info (Hamiltonian, operators, SOE params, eigenvalues/vectors).
*   **CalculationData**: Mutable container for results (Propagator, Green's function) stored as `DLRData`.
*   **SolverData**: Container combining `ModelData` and `CalculationData`.
*   **BackBone**: Represents a diagram as a linear chain of nodes and edges.
*   **HybridizationState**: Efficient integer encoding for hybridization lines (`NoHyb`, `KPlus`, `KMinus`).

### Representation Formats
Defined in `src/solver/Representation.jl`:
*   **DenseFormat**: Uses full dense matrices for operators and propagators.
*   **BlockDenseFormat**: Exploits particle number conservation for block-sparse storage.

Selected via `format = "dense"` or `format = "blockdense"` in the input file.

## File Structure

*   `src/QMBToolkitSOEHYB.jl`: Main module and `run` entry point.
*   `src/solver/`: Core logic (`solve.jl`, `Data.jl`, `Representation.jl`).
*   `src/diagram/`: Diagram generation and evaluation logic.
    *   `DiagramStructure.jl`: `BackBone` and `HybridizationState` definitions.
    *   `gen_*.jl`: Generators for propagator, Green's function, etc.
    *   `eval_*.jl`: Evaluators (some pending).
*   `src/util/`: I/O, initialization, and parameter handling.
*   `src/dense/`: Dense matrix operations implementation.
*   `src/blockdense/`: Block-dense operations implementation.
*   `bin/run_soehyb`: CLI wrapper script.
*   `examples/`: Sample input files and configurations.
*   `scripts/`: Utility scripts for generating input data.

## Dependencies

*   **Core Logic**: `QMBToolkitDLR`, `QMBToolkitED`, `LinearAlgebra`, `SparseArrays`, `QuadGK`.
*   **I/O**: `TOML`, `HDF5`, `DelimitedFiles`, `Printf`.
*   **Parallelism**: `MPI`.
*   **Progress**: `ProgressMeter`.
