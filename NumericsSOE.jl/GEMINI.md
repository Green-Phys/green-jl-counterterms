# Gemini Project Context: NumericsESPRIT.jl

## Project Overview

This is a Julia package named `NumericsESPRIT`. Based on the file structure and configuration files (`Project.toml`, `Manifest.toml`), it is a software project developed in the Julia programming language. The project is in its early stages, with placeholder source and test files. The purpose of the package is not yet clear from the code, but the name suggests it might be related to numerical algorithms, possibly the ESPRIT algorithm.

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
