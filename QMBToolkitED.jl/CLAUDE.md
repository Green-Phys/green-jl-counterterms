# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QMBToolkitED.jl is a Julia package for performing exact diagonalization (ED) on quantum many-body systems. The core workflow is:
1. Generate fermionic annihilation operators for a Fock space
2. Construct the Hamiltonian from hopping matrix and Coulomb tensor inputs
3. Solve (diagonalize) the Hamiltonian
4. Calculate observables: partition function Z, imaginary-time Green's function G(τ), and Matsubara Green's function G(iωₙ)

## Common Commands

### Testing
```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# From Julia REPL
using Pkg
Pkg.activate(".")
Pkg.test()
```

### Building Documentation
```bash
# Build documentation locally
julia --project=docs/ docs/make.jl
```

### Development Setup
```bash
# Activate the project environment
julia --project

# In Julia REPL, instantiate dependencies
using Pkg
Pkg.instantiate()
```

## Architecture

### Module Structure
The package is organized as a single module in `src/QMBToolkitED.jl` that exports six public functions:
- `get_annihilators(nflavors)` - Generates fermionic operators
- `build_hamiltonian(hopping_input, coulomb_input)` - Constructs Hamiltonian matrix
- `solve_hamiltonian(H)` - Diagonalizes the Hamiltonian
- `get_Z(evals, beta)` - Calculates partition function
- `calculate_green_tau(...)` - Computes G(τ)
- `calculate_green_matsubara(...)` - Computes G(iωₙ)

### Fermionic Operators
- Uses occupation number basis where the i-th bit of an integer represents the i-th state's occupation
- For `nflavors = 3`, state |100⟩ corresponds to integer 4
- Operators are stored as block-dense matrices for memory efficiency
- Sign tracking ensures proper fermionic anti-commutation relations

### Hamiltonian Construction
The Hamiltonian is built from two inputs:

**Hopping matrix** (FLAVORS×FLAVORS lines, 4 columns each):
- Columns: flavor i (0-indexed), flavor j (0-indexed), Re(H_ij), Im(H_ij)
- Implements: H = Σᵢⱼ Hᵢⱼ c†ᵢ cⱼ

**Coulomb tensor** (first line = count, then 7 columns per entry):
- Columns: element index, i, j, k, l (all 0-indexed), Re(U_ijkl), Im(U_ijkl)
- Implements: H = 0.5 Σᵢⱼₖₗ Uᵢⱼₖₗ c†ᵢ c†ⱼ cₖ cₗ
- No symmetrization needed; matrix construction follows definition literally

Both inputs can be file paths (String) or IO buffers (IOBuffer).

### Green's Function Calculations
Both functions transform operators to the Hamiltonian eigenbasis and shift energies so ground state = 0:
- **Imaginary-time G(τ)**: Uses trace formula with evolution operators
- **Matsubara G(iωₙ)**: Uses Lehmann representation with Boltzmann factors
- Matrix element cutoff: `abs(matrix_element) > 1e-9`

### Test Coverage
Tests in `test/runtests.jl` validate:
1. Anti-commutation relations {cᵢ, cⱼ} = 0 and {cᵢ, c†ⱼ} = δᵢⱼ
2. Numerical results for physical models vs analytical solutions:
   - Anderson Impurity Model (partition function)
   - Non-interacting model (G(τ) and G(iωₙ))
   - Hubbard atom (particle-hole symmetric at half-filling)
   - Hubbard dimer (both non-degenerate and near-degenerate ground states)

## Dependencies
- Julia 1.11.5 (standard library only: LinearAlgebra)
- Test dependencies: Test
- Documentation: Documenter.jl

## CI/CD
- GitHub Actions workflow `.github/workflows/CI.yml`
- Tests run on Julia 1.11, Ubuntu, x64
- Documentation auto-deploys from `main` branch
- Coverage reported to Codecov