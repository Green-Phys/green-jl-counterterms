# QMBToolkitED.jl

**Quantum Many-Body Toolkit: Exact Diagonalization**

A Julia package for exact diagonalization of quantum many-body Hamiltonians with two complementary methods:
- **Dense**: Full Hilbert space using dense matrices
- **BlockDense**: Symmetry-exploiting block-diagonal structure

## Features

- **Two Methods on Equal Footing**
  - Dense method: Traditional full Hilbert space approach
  - BlockDense method: Exploits quantum number conservation (10-100x memory reduction)

- **Fermionic Systems**
  - Automatic fermionic anti-commutation relations
  - Support for multi-orbital systems

- **Quantum Number Conservation**
  - Particle number (N)
  - Spin (N, Sz)
  - Site-based quantum number specification

- **Physical Observables**
  - Eigenvalues and eigenvectors
  - Partition functions
  - Imaginary-time Green's functions
  - Matsubara Green's functions

## Installation

```julia
using Pkg
Pkg.add("QMBToolkitED")
```

## Quick Start

### Dense Method (Full Hilbert Space)

```julia
using QMBToolkitED

# Define hopping and interaction
hopping_str = """
0 0 -0.5 0.0
1 1 -0.5 0.0
"""

coulomb_str = """
2
0 0 1 1 0 2.0 0.0
1 1 0 0 1 2.0 0.0
"""

# Build and solve Hamiltonian
H, c = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
evals, evecs = solve_hamiltonian(H)

# Calculate partition function
beta = 10.0
Z = get_Z(evals, beta, shift_energy=true)

# Calculate Green's functions
iwn, g_iwn = calculate_green_matsubara(evals, evecs, c, beta, 100, 1, 1)
tau, g_tau = calculate_green_tau(evals, evecs, c, beta, 100, 1, 1)
```

### BlockDense Method (Symmetry Exploitation)

```julia
using QMBToolkitED
using QMBToolkitED.BlockDense

# Specify quantum numbers for each orbital (site-based)
# Example: 2-site Hubbard model [↑, ↓, ↑, ↓]
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]

# Build basis with (N, Sz) conservation
basis = make_basis(site_qns)

# Build Hamiltonian using block structure
H, c = build_hamiltonian(
    IOBuffer(hopping_str),
    IOBuffer(coulomb_str);
    method=:blockdense,
    basis=basis
)

# Solve (automatically uses efficient block-diagonal solver)
evals, evecs = solve_hamiltonian(H)

# Same Green's function interface works with both methods
iwn, g_iwn = calculate_green_matsubara(evals, evecs, c, beta, 100, 1, 1)
```

### Conversion Between Methods

```julia
# Convert BlockDense operators to dense matrices
c_dense = to_dense(c_blockdense)  # BlockDenseOperator → Matrix{ComplexF64}
```

## Input Format

### Hopping Matrix
Each line: `i j Re(t_ij) Im(t_ij)` (0-indexed)
```
0 0 -0.5 0.0    # On-site energy at site 0
1 1 -0.5 0.0    # On-site energy at site 1
0 1 -1.0 0.0    # Hopping from site 0 to 1
1 0 -1.0 0.0    # Hopping from site 1 to 0 (Hermitian)
```

### Coulomb Interaction
First line: `nflavors`
Subsequent lines: `# i j k l Re(U_ijkl) Im(U_ijkl)` (0-indexed)
```
2
0 0 1 1 0 2.0 0.0    # U_0110 (interaction between sites 0 and 1)
1 1 0 0 1 2.0 0.0    # U_1001 (Hermitian conjugate)
```

## Package Structure

```
src/
├── Dense/                  # Dense method (full Hilbert space)
│   ├── Operators.jl        # Fermionic operators
│   ├── Hamiltonian.jl      # Hamiltonian construction
│   └── Diagonalization.jl  # Standard eigenvalue solver
├── BlockDense/             # Block-dense method (symmetry exploitation)
│   ├── Core/               # Quantum numbers, sectors, basis
│   ├── Operations/         # Operator algebra and conversions
│   └── HighLevel/          # Hamiltonian and efficient solvers
├── UnifiedInterface.jl     # Unified API for both methods
└── QMBToolkitED.jl         # Main module
```

## Key Concepts

### Dense vs BlockDense

| Feature | Dense | BlockDense |
|---------|-------|------------|
| **Hilbert Space** | Full 2^n states | Symmetry sectors only |
| **Memory** | O(4^n) | O(4^n / #sectors) |
| **Operators** | Matrix{ComplexF64} | BlockDenseOperator |
| **Use Case** | Small systems (n ≤ 12) | Larger systems with symmetries |
| **Speedup** | Baseline | 10-100x for large systems |

### Quantum Numbers

The BlockDense method requires specifying quantum numbers for each orbital:

```julia
# Particle number only (N conservation)
site_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]

# Particle number and spin (N, Sz conservation)
site_qns = [
    QN_NSz(1, 1),   # Site 1, spin-up
    QN_NSz(1, -1),  # Site 1, spin-down
    QN_NSz(1, 1),   # Site 2, spin-up
    QN_NSz(1, -1)   # Site 2, spin-down
]
```

## Examples

See `examples/` directory for:
- Hubbard atom and dimer
- Anderson impurity model
- Multi-orbital systems
- Comparison of Dense vs BlockDense performance

## Testing

```julia
using Pkg
Pkg.test("QMBToolkitED")
```

Test suite (214 tests total):

### Global API Tests
User-facing functionality tested with **both** Dense and BlockDense methods:
- Fermionic operators and anticommutation relations
- Hamiltonian construction (Anderson impurity, Hubbard atom/dimer)
- Green's functions (Matsubara and imaginary-time)
- Unified interface (method parameter API)

### BlockDense Implementation Tests
Internal implementation details specific to BlockDense:
- Quantum number arithmetic (QN_N, QN_NSz)
- Sector partitioning and basis construction
- Operator flux computation from site_qns
- Block-dense algebra and conversions

### Consistency Tests
Cross-validates that Dense and BlockDense produce identical results:
- Physical observables (eigenvalues, partition functions)
- Hamiltonians (non-interacting, Hubbard, Anderson impurity)

## Documentation

Full documentation available at: [https://QMBToolkit.github.io/QMBToolkitED.jl/dev/](https://QMBToolkit.github.io/QMBToolkitED.jl/dev/)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:
```bibtex
@software{qmbtoolkited,
  title = {QMBToolkitED.jl: Exact Diagonalization for Quantum Many-Body Systems},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/QMBToolkit/QMBToolkitED.jl}
}
```
