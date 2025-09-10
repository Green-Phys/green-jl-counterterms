# GEMINI.md: QMBToolkitED.jl

## Project Overview

This is a Julia project named `QMBToolkitED`. It provides a toolkit for performing exact diagonalization (ED) on quantum many-body systems.

The core functionality of this package includes:
- Generating fermionic annihilation operators.
- Solving the Hamiltonian to find its eigenvalues and eigenvectors.
- Calculating the imaginary-time Green's function.

The project is structured as a standard Julia package with source code in `src/`, tests in `test/`, and documentation in `docs/`. It depends on the `LinearAlgebra` and `SparseArrays` standard libraries.

## Input format
Two text file is needed to specify the Hamiltonian:

### hopping_matrix_input_file
Text file containing the hopping matrix (H_{ij}). The file must have FLAVORS*FLAVORS lines. Each line has four columns.

1st column: the flavor index i (from 0 to FLAVORS-1),
2nd column: the flavor index j (from 0 to FLAVORS-1),
3rd column: real part of hopping,
4th column: imaginary part of hopping.

The non-interacting part of the Hamiltonian reads H = \sum_{ij} H_{ij} c^\dagger_i c_j. The index j runs first.

### coulomb_tensor_input_file
Text file containing the Coulomb tensor U_{ijkl}. The Hamiltonian reads H = 0.5 \sum_{ijkl} U_{ijkl} c^\dagger_i c^\dagger_j c_k c_l. One does not have to symmetrize the tensor. The Hamiltonian matrix is constructed following the above definition "literally". 
The first line specifies the number of non-zero elements in the Coulomb tensor. From the second line, each line must contain seven columns.

1st column: the index of non-zero elements in the tensor (from 0 in increasing order),
2nd column: the flavor index i (from 0 to FLAVORS-1),
3rd column: the flavor index j (from 0 to FLAVORS-1),
4th column: the flavor index k (from 0 to FLAVORS-1),
5th column: the flavor index l (from 0 to FLAVORS-1),
6th column: real part of U_{ijkl},
7th column: imaginary part of U_{ijkl}

## Building and Running

This is a library package, so there is no main application to run. The functionality is intended to be used from other Julia code or an interactive Julia session.

### Testing

The project uses the standard `Test` library for unit testing. Tests can be run from the Julia REPL in the project's root directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

Alternatively, using the shell:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

### Documentation

The documentation is built using `Documenter.jl`. To build the documentation locally, run the following command from the project's root directory:

```sh
julia --project=docs/ docs/make.jl
```

## Development Conventions

### Code Style

- The main logic is contained within the `QMBToolkitED` module in `src/QMBToolkitED.jl`.
- Public functions are exported and documented with docstrings.
- The code leverages sparse matrices (`SparseMatrixCSC`) for efficient representation of quantum operators.

### Testing

- Tests are located in the `test/` directory, with `runtests.jl` as the main entry point.
- The tests cover both the fundamental algebraic properties of the fermionic operators and the numerical results of the Anderson Impurity Model, Non-interacting model, Hubbard atom, and Hubbard dimer.

### Dependencies

- Project dependencies are managed through the `Project.toml` and `Manifest.toml` files, which is standard for Julia projects.
- The CI pipeline defined in `.github/workflows/CI.yml` automatically runs tests and builds the documentation on pushes and pull requests to the `main` branch.
