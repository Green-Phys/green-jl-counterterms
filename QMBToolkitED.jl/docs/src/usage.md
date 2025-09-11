# Usage Example: Hubbard Dimer

This example demonstrates how to use `QMBToolkitED.jl` to calculate the Green's function for a Hubbard dimer at half-filling.

## 1. Define the Model

First, we define the parameters for our model. We'll consider the non-degenerate ground state case where the hopping parameter `t` is significant compared to the on-site interaction `U`.

```@example usage
using QMBToolkitED
t = 1.0  # Hopping strength
U = 2.0  # On-site interaction
beta = 20.0 # Inverse temperature
n_matsubara = 100 # Number of Matsubara frequencies
n_tau = 100 # Number of imaginary time points

# Flavors are defined as:
# 0: site 1, up
# 1: site 1, down
# 2: site 2, up
# 3: site 2, down
```

## 2. Construct the Hamiltonian

We create the hopping matrix and Coulomb tensor as strings. The model is defined at half-filling by setting the on-site energy to `-U/2`.

```@example usage
hopping_str = """
0 0 $(-U/2) 0.0
1 1 $(-U/2) 0.0
2 2 $(-U/2) 0.0
3 3 $(-U/2) 0.0
0 2 $(-t) 0.0
2 0 $(-t) 0.0
1 3 $(-t) 0.0
3 1 $(-t) 0.0
"""

coulomb_str = """
4
0 0 1 1 0 $(U) 0.0
1 1 0 0 1 $(U) 0.0
2 2 3 3 2 $(U) 0.0
3 3 2 2 3 $(U) 0.0
"""

# Build the Hamiltonian from the string inputs
H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str));
```

## 3. Solve the System

Next, we diagonalize the Hamiltonian to find its eigenvalues and eigenvectors.

```@example usage
evals, evecs = solve_hamiltonian(H);
```

## 4. Calculate Green's Functions

With the solution, we can now compute the Green's functions in both Matsubara frequency and imaginary time. We'll calculate the on-site (diagonal) Green's function `$G_{11}` and the inter-site (off-diagonal) Green's function `$G_{12}`.

```@example usage
# Matsubara Green's function G(iωₙ)
iwn, g11_matsubara = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1);
_,   g12_matsubara = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 3);

# Imaginary-time Green's function G(τ)
tau_grid, g11_tau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1);
_,        g12_tau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 3);
```

## 5. Plot the Results

Finally, we can visualize the results.

### Matsubara Green's Function

![Matsubara Green's Function](assets/matsubara_plot.png)

### Imaginary-Time Green's Function

![Imaginary-Time Green's Function](assets/tau_plot.png)
