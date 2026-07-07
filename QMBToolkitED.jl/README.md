# QMBToolkitED.jl

**Quantum Many-Body Toolkit: Exact Diagonalization**

A Julia package for exact diagonalization of quantum many-body Hamiltonians, exploiting quantum number conservation for block-sparse efficiency.

## Features

- **Block-Sparse Efficiency**
  - Operators stored as block-diagonal/block-sparse matrices grouped by symmetry sectors
  - Each sector diagonalized independently: O(n * d^3) instead of O((n*d)^3)
  - Graceful fallback: `QN_Trivial` recovers the dense (no symmetry) limit

- **Quantum Number Conservation**
  - `QN_Trivial` -- No symmetry (full Hilbert space, single sector)
  - `QN_N` -- Particle number conservation
  - `QN_NSz` -- Particle number and spin-z conservation
  - Flavor-based quantum number specification

- **Fermionic Systems**
  - Automatic fermionic anti-commutation relations via Jordan-Wigner strings
  - Multi-flavor support
  - Operator algebra: multiplication, addition, adjoint, scalar operations

- **Physical Observables**
  - Eigenvalues and eigenvectors
  - Partition function
  - Imaginary-time Green's function G(tau)
  - Matsubara Green's function G(i omega_n)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/QMBToolkit/QMBToolkitED.jl")
```

## Quick Start

```julia
using QMBToolkitED

# === 1. Define quantum numbers for each flavor ===
# Hubbard atom: 2 flavors (spin-up, spin-down)
flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]

# === 2. Build basis ===
basis = Basis(flavor_qns)

# === 3. Define Hamiltonian inputs ===
U = 2.0

hopping_str = """
0 0 $(-U/2) 0.0
1 1 $(-U/2) 0.0
"""

coulomb_str = """
2
0 0 1 1 0 $(U) 0.0
1 1 0 0 1 $(U) 0.0
"""

# === 4. Build and solve Hamiltonian ===
H, c = build_hamiltonian(basis, IOBuffer(hopping_str), IOBuffer(coulomb_str))
result = solve_hamiltonian(H)

# === 5. Compute observables ===
beta = 10.0
evals = to_vector(result.eigenvalues)
Z = get_Z(evals, beta; shift_energy=true)

# Matsubara Green's function G_11(i omega_n)
iwn, g_iwn = calculate_green_matsubara(result, c[1], c[1], beta, 100)

# Imaginary-time Green's function G_11(tau)
tau, g_tau = calculate_green_tau(result, c[1], c[1], beta, 100)
```

### Choosing Quantum Numbers

The choice of quantum number type controls how finely the Hilbert space is partitioned:

```julia
nflavors = 4

# No symmetry: 1 sector of dimension 2^4 = 16
flavor_qns = [QN_Trivial() for _ in 1:nflavors]

# Particle number conservation: sectors grouped by total N
flavor_qns = [QN_N(1) for _ in 1:nflavors]

# Particle number + spin conservation: sectors grouped by (N, Sz)
flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
```

More symmetry = more sectors = smaller blocks = faster diagonalization.

## Input Format

### Hopping Matrix

Each line: `i j Re(t_ij) Im(t_ij)` (0-indexed flavors)

Implements: H_hop = sum_ij t_ij c_i^dag c_j

```
0 0 -0.5 0.0    # Energy for flavor 0
1 1 -0.5 0.0    # Energy for flavor 1
0 1 -1.0 0.0    # Hopping from flavor 1 to 0
1 0 -1.0 0.0    # Hopping from flavor 0 to 1
```

### Coulomb Interaction

First line: number of non-zero entries.
Subsequent lines: `# i j k l Re(U_ijkl) Im(U_ijkl)` (0-indexed flavors)

Implements: H_int = 0.5 sum_ijkl U_ijkl c_i^dag c_j^dag c_k c_l

```
2
0 0 1 1 0 2.0 0.0
1 1 0 0 1 2.0 0.0
```

Both inputs accept file paths (`String`) or in-memory buffers (`IOBuffer`).

## Package Structure

```
src/
├── QMBToolkitED.jl    # Main module, exports, get_Z
├── QuantumNumbers.jl   # QN_Trivial, QN_N, QN_NSz types and arithmetic
├── Basis.jl            # Sector, Basis types; Basis constructor
├── Operators.jl        # Operator, DiagonalOperator; annihilator(s)
├── Algebra.jl          # Operator arithmetic (*, +, -, adjoint, scalar)
├── Conversion.jl       # to_matrix, to_vector
├── Hamiltonian.jl      # build_hamiltonian from hopping/Coulomb inputs
├── Diagonalization.jl  # solve_hamiltonian, EigenResult
└── Greens.jl           # calculate_green_tau, calculate_green_matsubara
```

## Key Types

| Type | Description |
|------|-------------|
| `QN_Trivial`, `QN_N`, `QN_NSz` | Quantum number types |
| `Sector` | States sharing the same quantum number |
| `Basis` | Hilbert space partitioned into sectors |
| `Operator` | Block-sparse matrix with quantum number flux |
| `DiagonalOperator` | Block-diagonal storage (eigenvalues) |
| `EigenResult` | Eigenvalues + eigenvectors from `solve_hamiltonian` |

## Testing

```julia
using Pkg
Pkg.test("QMBToolkitED")
```

Test suite (443 tests):

### Unit Tests (`test/unit/`)
- Quantum number types, arithmetic, hashing, ordering
- Basis construction and sector partitioning
- Operator construction, fermionic signs, block structure
- Operator algebra (multiplication, addition, adjoint, scalar)

### Physics Validation (`test/physics/`)
Each model is tested with all three quantum number types (`QN_Trivial`, `QN_N`, `QN_NSz`):
- **1D Chain**: Non-interacting, analytical G(tau) and G(i omega_n)
- **Anderson Impurity**: Partition function validation
- **Hubbard Atom**: Partition function, G(tau), G(i omega_n) against exact formulas
- **Hubbard Dimer**: Eigenvalue spectrum and Green's functions
- **Hubbard Dimer (Near-Degenerate)**: Stress test with near-degenerate ground state

## Dependencies

- Julia 1.11+ (standard library only: `LinearAlgebra`, `Printf`)
- Test dependencies: `Test`

## Documentation

Full documentation available at: [https://QMBToolkit.github.io/QMBToolkitED.jl/dev/](https://QMBToolkit.github.io/QMBToolkitED.jl/dev/)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:
```bibtex
@software{qmbtoolkited,
  title = {QMBToolkitED.jl: Exact Diagonalization for Quantum Many-Body Systems},
  author = {Yang Yu},
  year = {2025},
  url = {https://github.com/QMBToolkit/QMBToolkitED.jl}
}
```
