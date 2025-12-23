"""
Unified interface for Dense and BlockDense methods.

Provides a consistent API that works with both implementations, allowing users
to choose the backend via the `method` keyword argument.
"""

# ============================================================================
# Basis Construction
# ============================================================================

"""
    make_basis(nflavors::Int) -> DenseBasis
    make_basis(site_qns::Vector{<:QuantumNumber}) -> BlockDenseBasis

Unified interface for constructing a basis.

For dense method, pass the number of flavors:
```julia
basis = make_basis(3)  # Returns DenseBasis
```

For block-dense method, pass site quantum numbers:
```julia
basis = make_basis([QN_N(1), QN_N(1), QN_N(1)])  # Returns BlockDenseBasis
```

The method is automatically determined by the argument type.
"""
make_basis(nflavors::Int) = Dense.make_basis_dense(nflavors)
make_basis(site_qns::Vector{<:BlockDense.QuantumNumber}) = BlockDense.make_basis_blockdense(site_qns)

# ============================================================================
# Fermionic Operators
# ============================================================================

"""
    get_annihilators(nflavors::Int; method::Symbol=:dense)
    get_annihilators(basis::Dense.DenseBasis; method::Symbol=:dense)
    get_annihilators(basis::BlockDense.BlockDenseBasis; method::Symbol=:blockdense)

Generate fermionic annihilation operators.

Arguments:
- `nflavors::Int`: Number of orbitals (for dense method)
- `basis::DenseBasis`: Dense basis (for dense method)
- `basis::BlockDenseBasis`: Block-dense basis (for blockdense method)
- `method::Symbol`: `:dense` or `:blockdense`

Returns:
- For `:dense`: `Vector{DenseOperator}`
- For `:blockdense`: `Vector{BlockDenseOperator}`

Examples:
```julia
# Dense (default) - convenience method
c_dense = get_annihilators(3)

# Dense with explicit basis
basis = Dense.make_basis_dense(3)
c_dense = get_annihilators(basis)

# Block-dense
basis = BlockDense.make_basis([BlockDense.QN_N(1) for _ in 1:3])
c_block = get_annihilators(basis; method=:blockdense)
```
"""
function get_annihilators(nflavors::Int; method::Symbol=:dense)
    if method == :dense
        return Dense.get_annihilators_dense(nflavors)
    else
        error("For block-dense method, pass a BlockDenseBasis instead of nflavors. " *
              "Example: basis = BlockDense.make_basis([BlockDense.QN_N(1) for _ in 1:$nflavors]); " *
              "get_annihilators(basis)")
    end
end

# Dispatch for dense basis
function get_annihilators(basis::Dense.DenseBasis; method::Symbol=:dense)
    if method == :dense
        return Dense.get_annihilators_dense(basis)
    else
        error("Cannot use :blockdense method with DenseBasis. " *
              "Use get_annihilators(BlockDenseBasis) instead.")
    end
end

# Dispatch for block-dense
function get_annihilators(basis::BlockDense.BlockDenseBasis; method::Symbol=:blockdense)
    if method == :blockdense
        return BlockDense.get_annihilators_blockdense(basis)
    else
        error("Cannot use :dense method with BlockDenseBasis. " *
              "Use get_annihilators(nflavors; method=:dense) instead.")
    end
end

"""
    build_hamiltonian(
        hopping_input, coulomb_input;
        method::Symbol=:dense,
        basis::Union{Nothing,Dense.DenseBasis,BlockDense.BlockDenseBasis}=nothing
    )

Construct a Hamiltonian from input files.

Arguments:
- `hopping_input`: Path to hopping matrix file or IO
- `coulomb_input`: Path to Coulomb tensor file or IO
- `method::Symbol`: `:dense` or `:blockdense`
- `basis`: Required for `:blockdense` method, optional for `:dense`

Returns:
- For `:dense`: `(DenseOperator, Vector{DenseOperator})`
- For `:blockdense`: `(BlockDenseOperator, Vector{BlockDenseOperator})`

Examples:
```julia
# Dense (default)
H, c = build_hamiltonian("hopping.txt", "coulomb.txt")

# Block-dense
basis = BlockDense.make_basis([BlockDense.QN_N(1) for _ in 1:4])
H, c = build_hamiltonian("hopping.txt", "coulomb.txt"; method=:blockdense, basis=basis)
```
"""
function build_hamiltonian(
    hopping_matrix_input::Union{String, IO},
    coulomb_tensor_input::Union{String, IO};
    method::Symbol=:dense,
    basis::Union{Nothing,Dense.DenseBasis,BlockDense.BlockDenseBasis}=nothing
)
    if method == :dense
        return Dense.build_hamiltonian_dense(hopping_matrix_input, coulomb_tensor_input)
    elseif method == :blockdense
        if basis === nothing
            error("basis argument is required for :blockdense method. " *
                  "Create a basis using BlockDense.make_basis([BlockDense.QN_N(1), ...])")
        end
        if !(basis isa BlockDense.BlockDenseBasis)
            error("basis must be a BlockDenseBasis for :blockdense method")
        end
        return BlockDense.build_hamiltonian_blockdense(basis, hopping_matrix_input, coulomb_tensor_input)
    else
        error("Unknown method: $method. Use :dense or :blockdense")
    end
end

"""
    solve_hamiltonian(H; method::Symbol=:auto)

Diagonalize a Hamiltonian.

Arguments:
- `H`: Hamiltonian (DenseOperator or BlockDenseOperator)
- `method::Symbol`: `:auto` (detect from type), `:dense`, or `:blockdense`

Returns:
- `DenseEigenResult` or `BlockEigenResult` containing:
  - `eigenvalues`: Diagonal operator with eigenvalues
  - `eigenvectors`: Operator with eigenvector matrix

Examples:
```julia
# Dense with DenseOperator
H, c = build_hamiltonian("hopping.txt", "coulomb.txt")
result = solve_hamiltonian(H)

# Access eigenvalues and eigenvectors
evals = result.eigenvalues.diag
evecs = result.eigenvectors.data

# Block-dense
basis = BlockDense.make_basis([BlockDense.QN_N(1) for _ in 1:4])
H, c = build_hamiltonian("hopping.txt", "coulomb.txt"; method=:blockdense, basis=basis)
result = solve_hamiltonian(H)

# Access eigenvalues per sector
for (idx, evals) in result.eigenvalues.sector_diags
    println("Sector ", idx, ": ", evals)
end
```
"""
function solve_hamiltonian(H::Dense.DenseOperator; method::Symbol=:auto)
    if method == :auto || method == :dense
        return Dense.solve_hamiltonian_dense(H)
    else
        error("Hamiltonian is DenseOperator, use method=:dense or method=:auto")
    end
end


function solve_hamiltonian(H::BlockDense.BlockDenseOperator; method::Symbol=:auto)
    if method == :auto || method == :blockdense
        return BlockDense.solve_hamiltonian_blockdense(H)
    else
        error("Hamiltonian is BlockDenseOperator, use method=:blockdense or method=:auto")
    end
end

"""
    calculate_green_tau(result, annh_ops, beta, n_tau, i, j)

Unified interface for calculating imaginary-time Green's function.
Dispatches to correct backend based on result type.
"""
function calculate_green_tau(
    result::Dense.DenseEigenResult,
    annh_ops::Vector{Dense.DenseOperator},
    beta::Real,
    n_tau::Int,
    i::Int,
    j::Int
)
    return Dense.calculate_green_tau_dense(result, annh_ops[i], annh_ops[j], beta, n_tau)
end

function calculate_green_tau(
    result::BlockDense.BlockEigenResult,
    annh_ops::Vector{BlockDense.BlockDenseOperator},
    beta::Real,
    n_tau::Int,
    i::Int,
    j::Int
)
    return BlockDense.calculate_green_tau_blockdense(result, annh_ops[i], annh_ops[j], beta, n_tau)
end


"""
    calculate_green_matsubara(result, annh_ops, beta, n_matsubara, i, j)

Unified interface for calculating Matsubara Green's function.
"""
function calculate_green_matsubara(
    result::Dense.DenseEigenResult,
    annh_ops::Vector{Dense.DenseOperator},
    beta::Real,
    n_matsubara::Int,
    i::Int,
    j::Int
)
    return Dense.calculate_green_matsubara_dense(result, annh_ops[i], annh_ops[j], beta, n_matsubara)
end

function calculate_green_matsubara(
    result::BlockDense.BlockEigenResult,
    annh_ops::Vector{BlockDense.BlockDenseOperator},
    beta::Real,
    n_matsubara::Int,
    i::Int,
    j::Int
)
    return BlockDense.calculate_green_matsubara_blockdense(result, annh_ops[i], annh_ops[j], beta, n_matsubara)
end


"""
    get_Z(evals::Vector{<:Real}, beta::Real; shift_energy::Bool=false)

Calculate the partition function Z = sum(exp(-beta * evals)).
If `shift_energy` is true, the ground state energy is shifted to zero before calculation.
"""
function get_Z(evals::Vector{<:Real}, beta::Real; shift_energy::Bool=false)
    local_evals = copy(evals)
    if shift_energy
        local_evals .-= minimum(local_evals)
    end
    return sum(exp.(-beta .* local_evals))
end

# ============================================================================
# Conversion Functions
# ============================================================================

"""
    to_matrix(op) -> Matrix{ComplexF64}

Unified interface for converting an operator to a dense matrix.

Works with:
- `DenseOperator` → extracts underlying matrix
- `BlockDenseOperator` → assembles blocks into full matrix
- `DenseDiagonalOperator` → converts to full diagonal matrix
- `BlockDenseDiagonalOperator` → converts to full diagonal matrix

Example:
```julia
# Dense operator
c = get_annihilators(3)
mat = to_matrix(c[1])

# Block-dense operator
basis = make_basis([QN_N(1), QN_N(1), QN_N(1)])
c_block = get_annihilators(basis)
mat_block = to_matrix(c_block[1])
```
"""
to_matrix(op::Dense.DenseOperator) = Dense.to_matrix_dense(op)
to_matrix(op::Dense.DenseDiagonalOperator) = Dense.to_matrix_dense(op)
to_matrix(op::BlockDense.BlockDenseOperator) = BlockDense.to_matrix_blockdense(op)
to_matrix(op::BlockDense.BlockDenseDiagonalOperator) = BlockDense.to_matrix_blockdense(op)

"""
    to_vector(D) -> Vector{Float64}

Unified interface for extracting diagonal values from a diagonal operator.

Works with:
- `DenseDiagonalOperator` → returns diagonal elements
- `BlockDenseDiagonalOperator` → returns concatenated diagonal elements in sector order

Example:
```julia
# Dense eigenvalues
result = solve_hamiltonian(H)
evals = to_vector(result.eigenvalues)

# Block-dense eigenvalues
result_block = solve_hamiltonian(H_block)
evals_block = to_vector(result_block.eigenvalues)
```
"""
to_vector(D::Dense.DenseDiagonalOperator) = Dense.to_vector_dense(D)
to_vector(D::BlockDense.BlockDenseDiagonalOperator) = BlockDense.to_vector_blockdense(D)
