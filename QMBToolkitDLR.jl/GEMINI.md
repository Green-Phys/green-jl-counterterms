# Project Overview

This project, `QMBToolkitDLR.jl`, is a Julia package that provides functions for performing numerical operations in the Discrete Lehmann Representation (DLR). The DLR is a compact basis for representing single-particle Green's functions and other imaginary-time correlation functions in quantum many-body physics.

The core functionality of this package includes:
- `do_convolution_dlr`: Performs convolution of two functions represented in the DLR basis.
- `do_reflection_dlr`: Performs a reflection operation `G(tau) = H(beta - tau)`.
- `do_multiplication_dlr`: Performs element-wise multiplication of two functions in the DLR basis.

The project uses the `Lehmann.jl` package for the underlying DLR implementation.

# Building and Running

This is a Julia project. To use it, you would typically add it to your Julia project's dependencies.

## Running Tests

To run the tests for this project, you can use the Julia package manager:

```julia
using Pkg
Pkg.test("QMBToolkitDLR")
```

Alternatively, from the shell, you can run:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

# Development Conventions

The code is written in Julia. The tests in the `test/` directory demonstrate how to use the functions in this package. The tests construct analytical functions, convert them to the DLR basis, perform the operations, and then convert back to the time domain to compare with the analytical result. This is a good pattern to follow for adding new tests.
