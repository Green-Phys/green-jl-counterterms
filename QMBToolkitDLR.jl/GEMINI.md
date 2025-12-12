# Project Overview

`QMBToolkitDLR.jl` is a Julia package providing high-performance numerical operations in the Discrete Lehmann Representation (DLR) for quantum many-body physics. It builds upon the `Lehmann.jl` package and extends it with optimized convolution and reflection routines, supporting both standard dense arrays and block-sparse structures (via `QMBToolkitED.jl`).

## Core Functionality

The package offers two main modes of operation:

### 1. Dense Operations (`src/dense/`)
Designed for standard dense tensors (e.g., Green's functions `G[τ, i, j]`).
- **Convolution**: `do_convolution_dlr2tau!`
  - Computes `H(τ) = ∫ dτ' F(τ-τ') G(τ')`.
  - Optimized with pre-allocated workspaces and kernel caching to minimize allocations.
- **Reflection**: `do_reflection_tau2tau`
  - Performs `G(τ) => G(β - τ)`.

### 2. BlockDense Operations (`src/blockdense/`)
Designed for systems with symmetries (Quantum Numbers), where operators are block-diagonal in specific sectors.
- **Data Structure**: `BlockDenseArray` (from `QMBToolkitED` context)
  - Handles sector-specific data (e.g., N=0, N=1 blocks).
- **Convolution**: `do_convolution_dlr2tau_blockdense!`
- **Reflection**: `do_reflection_blockdense`
- **Conversions**: `tau2dlr_blockdense`, `dlr2tau_blockdense`

## Key Structures

### Workspace & Caching (`src/workspace.jl`)
To achieve high performance, especially in inner loops, the package uses:
- **`ConvolutionWorkspace`**: Pre-allocates buffers for intermediate calculations, saving GBs of allocations in heavy workloads.
- **`KernelCache`**: Pre-computes DLR kernel evaluations `K(0, ω, β)`.

## Dependencies

- **`Lehmann.jl`**: Core DLR basis implementation.
- **`QMBToolkitED.jl`**: Provides the `BlockDense` infrastructure and quantum number handling.

# Building and Running

## Installation
Add the package to your Julia environment:
```julia
using Pkg
Pkg.add(path=".") # For local development
```

## Running Tests
The test suite covers both dense and BlockDense operations, comparing numerical results against analytical solutions.

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Or from the Julia REPL:
```julia
using Pkg
Pkg.test("QMBToolkitDLR")
```

# Development Conventions

- **Code Structure**:
  - `src/dense`: Standard array implementations.
  - `src/blockdense`: Block-sparse implementations using `QMBToolkitED` types.
  - `src/workspace.jl`: Shared types and performance utilities.
- **Testing**:
  - Tests verify numerical accuracy against analytical functions (e.g., simple poles).
  - New features should include similar analytical verifications.