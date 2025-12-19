# Project Overview

This is a Julia project that implements a bold hybridization expansion solver based on the sum of exponentials. The project is in its early stages of development.

## Building and Running

This is a Julia project. The following commands can be used to work with it:

*   **Instantiate the environment:**
    ```julia
    using Pkg
    Pkg.instantiate()
    ```

*   **Run the tests:**
    ```julia
    using Pkg
    Pkg.test()
    ```

*   **Load the module:**
    ```julia
    using QMBToolkitSOEHYB
    ```

*   **Run the solver:**
    The main entry point of the solver is the `run` function, which takes an input file as an argument.
    ```julia
    using QMBToolkitSOEHYB
    run("path/to/your/input.toml")
    ```

## Development Conventions

*   The project follows the standard Julia package structure.
*   Source code is located in the `src` directory.
*   Tests are located in the `test` directory.
*   The main project file is `src/QMBToolkitSOEHYB.jl`.
*   The project uses `Test.jl` for testing.
*   The main entry point is the `run` function in `src/QMBToolkitSOEHYB.jl`.
*   The core solver logic is located in `src/solver/solve.jl`.
*   Diagram generation and evaluation logic is in the `src/diagram` directory.
*   Utility functions are in the `src/util` directory.

## Dependencies

The project has the following dependencies:

*   DelimitedFiles
*   HDF5
*   LinearAlgebra
*   Printf
*   ProgressMeter
*   QMBToolkitDLR
*   QMBToolkitED
*   QuadGK
*   SparseArrays
*   TOML