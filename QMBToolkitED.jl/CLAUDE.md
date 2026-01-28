# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QMBToolkitED.jl is a Julia package for performing exact diagonalization (ED) on quantum many-body systems. The core workflow is:
1. Create a basis with explicit quantum numbers (QN_Trivial, QN_N, or QN_NSz)
2. Generate fermionic annihilation operators for the Fock space
3. Construct the Hamiltonian from hopping matrix and Coulomb tensor inputs
4. Solve (diagonalize) the Hamiltonian
5. Calculate observables: partition function Z, imaginary-time Green's function G(τ), and Matsubara Green's function G(iωₙ)

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
The package exploits quantum number conservation for block-sparse efficiency. All source files live directly under `src/`.

**Quantum Number Types:**
- `QN_Trivial` - No symmetry (equivalent to dense representation)
- `QN_N` - Particle number conservation only
- `QN_NSz` - Particle number and spin-z conservation

**Main API:**
```julia
using QMBToolkitED

# Create basis with quantum numbers
basis = make_basis([QN_Trivial() for _ in 1:nflavors])  # No symmetry
basis = make_basis([QN_N(1) for _ in 1:nflavors])       # N conservation
basis = make_basis([QN_NSz(1, 1), QN_NSz(1, -1), ...])  # N+Sz conservation

# Generate operators and build Hamiltonian
c = get_annihilators(basis)
H, c = build_hamiltonian(basis, hopping_input, coulomb_input)

# Solve and compute observables
result = solve_hamiltonian(H)
evals = to_vector(result.eigenvalues)
Z = get_Z(evals, beta; shift_energy=true)
iwn, G = calculate_green_matsubara(result, c[i], c[j], beta, n_matsubara)
tau, G = calculate_green_tau(result, c[i], c[j], beta, n_tau)
```

### Source Layout (`src/`)
- `QuantumNumbers.jl` - QN_Trivial, QN_N, QN_NSz types and arithmetic
- `Basis.jl` - Sector, Basis types; make_basis; utility functions
- `Operators.jl` - Operator, DiagonalOperator types; get_annihilator(s)
- `Algebra.jl` - Operator arithmetic (*, +, -, adjoint, scalar)
- `Conversion.jl` - to_matrix, to_vector
- `Hamiltonian.jl` - build_hamiltonian from hopping/Coulomb inputs
- `Diagonalization.jl` - solve_hamiltonian, EigenResult type
- `Greens.jl` - calculate_green_tau, calculate_green_matsubara

### Fermionic Operators
- Uses occupation number basis where the i-th bit of an integer represents the i-th state's occupation
- For `nflavors = 3`, state |100⟩ corresponds to integer 4
- Operators are stored as block-sparse matrices grouped by quantum number sectors
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
Tests are organized under `test/`:
- `physics/` - Physics validation against analytical results:
  - Anti-commutation relations {cᵢ, cⱼ} = 0 and {cᵢ, c†ⱼ} = δᵢⱼ
  - Anderson Impurity Model (partition function)
  - Non-interacting model (G(τ) and G(iωₙ))
  - Hubbard atom (particle-hole symmetric at half-filling)
  - Hubbard dimer (both non-degenerate and near-degenerate ground states)
  - All tests run with QN_Trivial, QN_N, and QN_NSz
- `unit/` - Unit tests for types, basis construction, operators, and algebra

## Dependencies
- Julia 1.11.5 (standard library only: LinearAlgebra)
- Test dependencies: Test
- Documentation: Documenter.jl

## CI/CD
- GitHub Actions workflow `.github/workflows/CI.yml`
- Tests run on Julia 1.11, Ubuntu, x64
- Documentation auto-deploys from `main` branch
- Coverage reported to Codecov
