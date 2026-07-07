# Gemini Project Context: NumericsSOE.jl

## Project Overview

This is a Julia package named `NumericsSOE`. Based on the file structure and configuration files (`Project.toml`, `Manifest.toml`), it is a software project developed in the Julia programming language. The purpose of the package is to solve the Sum-of-Exponentials problem, which involves finding the exponents and coefficients of a signal that is modeled as a sum of complex exponentials.

The package provides several methods for finding the exponents, such as ESPRIT and Prony's method, and methods for finding the coefficients, such as least squares and pseudo-inverse.

## Building and Running

### Testing

The project is set up with tests in the `test/` directory. To run the tests, you can use the Julia package manager.

```julia
# From the Julia REPL, with the project activated
using Pkg
Pkg.test()
```

Alternatively, from the shell:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### Documentation

The project uses `Documenter.jl` to build documentation. The documentation source is in the `docs/` directory.

To build the documentation locally, run the following command from the project root in the shell:

```bash
julia --project=docs/ docs/make.jl
```

This will generate the documentation in the `docs/build/` directory.

## Development Conventions

*   **Code Style:** No specific code style is enforced yet, but the standard Julia conventions are expected to be followed.
*   **Testing:** Tests are written using the `Test` standard library. The main test set is in `test/runtests.jl`.
*   **Dependencies:** Project dependencies are managed through `Project.toml` and `Manifest.toml`.
*   **Continuous Integration:** The project uses GitHub Actions for CI. The workflow in `.github/workflows/CI.yml` defines jobs for running tests across different Julia versions and for building and deploying documentation.