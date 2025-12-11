"""
Unified interface for Dense and BlockDense methods.

Provides a consistent API that works with both implementations, allowing users
to choose the backend via the `method` keyword argument.
"""

# Helper to get standard (evals, evecs) from any result
function get_result_tuple(res::Dense.DenseEigenResult)
    return Dense.get_eigenvalues(res), Dense.get_eigenvectors(res)
end

function get_result_tuple(res::BlockDense.BlockEigenResult)
    evals = BlockDense.get_full_eigenvalues(res)
    evecs = BlockDense.get_full_eigenvectors(res)
    perm = sortperm(evals)
    return evals[perm], evecs[:, perm]
end

# Make results iterable to support destructuring: evals, evecs = result
Base.iterate(res::Dense.DenseEigenResult) = Base.iterate(get_result_tuple(res))
Base.iterate(res::Dense.DenseEigenResult, state) = Base.iterate(get_result_tuple(res), state)
Base.length(::Dense.DenseEigenResult) = 2
Base.eltype(::Type{Dense.DenseEigenResult}) = Any

# Explicitly add iteration for BlockEigenResult (duck typing)
Base.iterate(res::BlockDense.BlockEigenResult) = Base.iterate(get_result_tuple(res))
Base.iterate(res::BlockDense.BlockEigenResult, state) = Base.iterate(get_result_tuple(res), state)
Base.length(::BlockDense.BlockEigenResult) = 2
Base.eltype(::Type{BlockDense.BlockEigenResult}) = Any

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
- `AbstractSolverResult`: A result object (`DenseEigenResult` or `BlockEigenResult`).
  This object is iterable, allowing destructuring into `(evals, evecs)` for backward compatibility.

  - `evals::Vector{Float64}`: Eigenvalues
  - `evecs::Matrix{ComplexF64}`: Eigenvectors

Examples:
```julia
# Dense
H, c = build_hamiltonian("hopping.txt", "coulomb.txt")
evals, evecs = solve_hamiltonian(H)  # Auto-detects dense

# Block-dense
basis = BlockDense.make_basis([BlockDense.QN_N(1) for _ in 1:4])
H, c = build_hamiltonian("hopping.txt", "coulomb.txt"; method=:blockdense, basis=basis)
result = solve_hamiltonian(H)  # Returns BlockEigenResult
evals, evecs = result          # Destructuring works (less efficient for Green's functions)
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
        # This returns a BlockEigenResult
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
    annh_ops::Vector{Matrix{ComplexF64}},
    beta::Real,
    n_tau::Int,
    i::Int,
    j::Int
)
    evals = Dense.get_eigenvalues(result)
    evecs = Dense.get_eigenvectors(result)
    return Dense.calculate_green_tau_dense(evals, evecs, annh_ops, beta, n_tau, i, j)
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

# Backward compatibility for tuple input (Dense only)
function calculate_green_tau(
    evals::Vector{Float64},
    evecs::Matrix{ComplexF64},
    annh_ops::Vector{Matrix{ComplexF64}},
    beta::Real,
    n_tau::Int,
    i::Int,
    j::Int
)
    return Dense.calculate_green_tau_dense(evals, evecs, annh_ops, beta, n_tau, i, j)
end

"""
    calculate_green_matsubara(result, annh_ops, beta, n_matsubara, i, j)

Unified interface for calculating Matsubara Green's function.
"""
function calculate_green_matsubara(
    result::Dense.DenseEigenResult,
    annh_ops::Vector{Matrix{ComplexF64}},
    beta::Real,
    n_matsubara::Int,
    i::Int,
    j::Int
)
    evals = Dense.get_eigenvalues(result)
    evecs = Dense.get_eigenvectors(result)
    return Dense.calculate_green_matsubara_dense(evals, evecs, annh_ops, beta, n_matsubara, i, j)
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

# Backward compatibility for tuple input (Dense only)
function calculate_green_matsubara(
    evals::Vector{<:Real},
    evecs::Matrix,
    annh_ops::Vector{Matrix{ComplexF64}},
    beta::Real,
    n_matsubara::Int,
    i::Int,
    j::Int
)
    return Dense.calculate_green_matsubara_dense(evals, evecs, annh_ops, beta, n_matsubara, i, j)
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