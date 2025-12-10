"""
Unified interface for Dense and BlockDense methods.

Provides a consistent API that works with both implementations, allowing users
to choose the backend via the `method` keyword argument.

Example:
```julia
# Dense (default, backwards compatible)
c = get_annihilators(3)

# Block-dense (requires basis)
basis = BlockDense.make_basis([BlockDense.QN_N(1) for _ in 1:3])
c = get_annihilators(basis; method=:blockdense)
```
"""

"""
    get_annihilators(nflavors::Int; method::Symbol=:dense)
    get_annihilators(basis::BlockDense.BlockDenseBasis; method::Symbol=:blockdense)

Generate fermionic annihilation operators.

Arguments:
- `nflavors::Int`: Number of orbitals (for dense method)
- `basis::BlockDenseBasis`: Block-sparse basis (for blockdense method)
- `method::Symbol`: `:dense` or `:blockdense`

Returns:
- For `:dense`: `Vector{Matrix{ComplexF64}}`
- For `:blockdense`: `Vector{BlockDenseOperator}`

Examples:
```julia
# Dense (default)
c_dense = get_annihilators(3)

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
        basis::Union{Nothing,BlockDense.BlockDenseBasis}=nothing
    )

Construct a Hamiltonian from input files.

Arguments:
- `hopping_input`: Path to hopping matrix file or IO
- `coulomb_input`: Path to Coulomb tensor file or IO
- `method::Symbol`: `:dense` or `:blockdense`
- `basis::BlockDenseBasis`: Required for `:blockdense` method

Returns:
- For `:dense`: `(Matrix{ComplexF64}, Vector{Matrix{ComplexF64}})`
- For `:blockdense`: `(BlockDenseOperator, Vector{BlockDenseOperator})`

Examples:
```julia
# Dense (default, backwards compatible)
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
    basis::Union{Nothing,BlockDense.BlockDenseBasis}=nothing
)
    if method == :dense
        return Dense.build_hamiltonian_dense(hopping_matrix_input, coulomb_tensor_input)
    elseif method == :blockdense
        if basis === nothing
            error("basis argument is required for :blockdense method. " *
                  "Create a basis using BlockDense.make_basis([BlockDense.QN_N(1), ...])")
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
- `H`: Hamiltonian (Matrix or BlockDenseOperator)
- `method::Symbol`: `:auto` (detect from type), `:dense`, or `:blockdense`

Returns:
- `(eigenvalues::Vector{Float64}, eigenvectors::Matrix{ComplexF64})`

The return format is the same for both methods (for compatibility).

Examples:
```julia
# Dense
H, c = build_hamiltonian("hopping.txt", "coulomb.txt")
evals, evecs = solve_hamiltonian(H)  # Auto-detects dense

# Block-dense
basis = BlockDense.make_basis([BlockDense.QN_N(1) for _ in 1:4])
H, c = build_hamiltonian("hopping.txt", "coulomb.txt"; method=:blockdense, basis=basis)
evals, evecs = solve_hamiltonian(H)  # Auto-detects blockdense
```
"""
function solve_hamiltonian(H::Union{Matrix, SparseMatrixCSC}; method::Symbol=:auto)
    if method == :auto || method == :dense
        return Dense.solve_hamiltonian_dense(H)
    else
        error("Hamiltonian is Matrix/SparseMatrixCSC, use method=:dense or method=:auto")
    end
end

function solve_hamiltonian(H::BlockDense.BlockDenseOperator; method::Symbol=:auto)
    if method == :auto || method == :blockdense
        return BlockDense.solve_hamiltonian_blockdense(H)
    else
        error("Hamiltonian is BlockDenseOperator, use method=:blockdense or method=:auto")
    end
end
