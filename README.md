# green-jl-counterterms

Source code accompanying the counterterm (hybrid Hamiltonian-diagrammatic) quantum impurity solver:

> Y. Yu, G. Harsha, L. Zhang, A. Jażdżewska, D. Zgid, X. Dong, E. Gull,
> *Hybrid Hamiltonian-diagrammatic quantum impurity solver*,
> [arXiv:2606.11095](https://arxiv.org/abs/2606.11095) (2026).

## Repository layout

| Package | Purpose |
|---|---|
| `QMBToolkitSOEHYB.jl` | Counterterm hybridization-expansion solver; the scripts used below live in `QMBToolkitSOEHYB.jl/scripts/` |
| `QMBToolkitED.jl` | Exact diagonalization of the impurity plus counterterm Hamiltonian |
| `QMBToolkitDLR.jl` | Discrete Lehmann representation utilities |
| `NumericsSOE.jl` | Sum-of-exponentials fitting (ESPRIT) for hybridization functions and counterterms |
| `Lehmann.jl` | DLR basis and analytic test hybridizations (e.g. semicircular bath) |
| `Tutorial/` | LaTeX notes with the full benchmark instructions and reference results |

The packages refer to each other by relative path, so keep them together in one checkout.

## Requirements

Julia 1.12 (the committed manifests were resolved with Julia 1.12). MPI.jl and HDF5.jl bring their own
binaries; no system MPI or HDF5 is needed.

## Quick start: spinless semicircular-bath benchmark

This reproduces the simplest benchmark from the paper: a non-interacting spinless impurity coupled to a
semicircular bath at $\beta t = 8$, with three counterterms and a bare hybridization expansion through $k = 2$.
It runs in under a minute on a single core.

```sh
git clone https://github.com/Green-Phys/green-jl-counterterms.git
cd green-jl-counterterms/QMBToolkitSOEHYB.jl

# one-time setup: the input generators also need ArgParse and NumericsSOE
julia --project=. -e 'using Pkg; Pkg.add("ArgParse"); Pkg.develop(path="../NumericsSOE.jl"); Pkg.instantiate()'

RUN=paper_inputs/spinless_beta8_nct3_k2
mkdir -p $RUN/data $RUN/output

# 1. impurity Hamiltonian (hopping.txt, Uijkl.txt)
julia --project=. scripts/gen_interaction_hopping.jl \
  --n_spin_impurity 1 --n_orbital_impurity 1 \
  --t 0.0 --U 0.0 --J 0.0 --Uprime 0.0 --mu 0.0 \
  --output_dir $RUN/data

# 2. fit the hybridization, add three counterterms, write the residual hybridization
julia --project=. scripts/gen_soe_counterterms_hyb_light.jl \
  --beta 8.0 --n_spin_impurity 1 --n_orbital_impurity 1 --n_tau 1000 \
  --hyb_type SemiCircular --hopping_file $RUN/data/hopping.txt \
  --n_soe_counterterms 3 --n_soe_hybridization 10 \
  --output_dir $RUN/data --do_subtraction_trick true

# 3. run the solver; the result is written to $RUN/output/sim.h5
julia --project=. -e 'using QMBToolkitSOEHYB; QMBToolkitSOEHYB.run(ARGS[1])' $RUN/input.toml
```

The solver input [`paper_inputs/spinless_beta8_nct3_k2/input.toml`](QMBToolkitSOEHYB.jl/paper_inputs/spinless_beta8_nct3_k2/input.toml)
is part of the repository. The output file `sim.h5` contains

- `/green/linear_grid/tau` and `/green/linear_grid/green_tau`: $G(\tau)$ on a linear grid,
- `/partition_function/partition_function_by_order`: partition-function estimates order by order,

and should give $G(0) \approx -0.5$ and $Z \approx 1.1274$ at every order.

For the interacting spinful Hubbard scan, the comparison to the analytic result, and reference numbers, see
[`Tutorial/CounterTermInstructions.tex`](Tutorial/CounterTermInstructions.tex).

## Tests

Each package has its own test suite, for example

```sh
cd QMBToolkitSOEHYB.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```
