"""
Block-sparse interface matching QMBToolkitED.jl API.

This module provides block-sparse implementations of the main QMBToolkitED.jl functions,
maintaining API compatibility for easy comparison and validation.

Main functions:
- get_annihilators_blocksparse: Generate fermionic operators (block-sparse)
- build_hamiltonian_blocksparse: Construct Hamiltonian from input files
"""

"""
    get_annihilators_blocksparse(basis::BlockSparseBasis) -> Vector{BlockSparseOperator}

Generate fermionic annihilation operators for a given block-sparse basis.

This is the block-sparse equivalent of `get_annihilators(nflavors)`.

Arguments:
- `basis::BlockSparseBasis`: The many-body basis with sector structure

Returns:
- `Vector{BlockSparseOperator}`: Array of annihilation operators c[i] for i=1:nflavors

Example:
```julia
# Create basis for 3-orbital system with N conservation
site_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = make_basis(site_qns)

# Get annihilation operators
c = get_annihilators_blocksparse(basis)

# c[i] is the annihilation operator for flavor i
```

Comparison with sparse version:
```julia
# Sparse version
c_sparse = get_annihilators(3)

# Block-sparse version
basis = make_basis([QN_N(1), QN_N(1), QN_N(1)])
c_block = get_annihilators_blocksparse(basis)

# Should match when converted to sparse
for i in 1:3
    @assert to_sparse(c_block[i]) ≈ c_sparse[i]
end
```
"""
function get_annihilators_blocksparse(basis::BlockSparseBasis)
    return [get_annihilator_blocksparse(basis, i) for i in 1:basis.nflavors]
end

"""
    build_hamiltonian_blocksparse(
        basis::BlockSparseBasis,
        hopping_matrix_input::Union{String, IO},
        coulomb_tensor_input::Union{String, IO}
    ) -> Tuple{BlockSparseOperator, Vector{BlockSparseOperator}}

Construct a block-sparse Hamiltonian from hopping and Coulomb interaction inputs.

This is the block-sparse equivalent of `build_hamiltonian(hopping, coulomb)`.

The Hamiltonian has the form:
    H = Σ_{ij} t_{ij} c_i' c_j + 0.5 Σ_{ijkl} U_{ijkl} c_i' c_j' c_k c_l

Arguments:
- `basis::BlockSparseBasis`: The many-body basis (must match nflavors in input files)
- `hopping_matrix_input`: Path to file or IO containing hopping matrix
  Format: Each line is "i j Re(t_ij) Im(t_ij)" (0-indexed)
- `coulomb_tensor_input`: Path to file or IO containing Coulomb tensor
  Format: First line is nflavors, then "# i j k l Re(U_ijkl) Im(U_ijkl)" (0-indexed)

Returns:
- `BlockSparseOperator`: The total Hamiltonian H
- `Vector{BlockSparseOperator}`: Annihilation operators used to build H

Note: The Hamiltonian should have flux = 0 (number conserving).

Example:
```julia
site_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = make_basis(site_qns)

H, c = build_hamiltonian_blocksparse(basis, "hopping.txt", "coulomb.txt")

# H should be Hermitian
H_sparse = to_sparse(H)
@assert H_sparse ≈ H_sparse' atol=1e-14

# H should have zero flux (number conserving)
@assert H.flux.delta_qn == QN_N(0)
```

Comparison with sparse version:
```julia
# Sparse version
H_sparse, c_sparse = build_hamiltonian("hopping.txt", "coulomb.txt")

# Block-sparse version
basis = make_basis([QN_N(1) for _ in 1:nflavors])
H_block, c_block = build_hamiltonian_blocksparse(basis, "hopping.txt", "coulomb.txt")

# Hamiltonians should match
@assert to_sparse(H_block) ≈ H_sparse atol=1e-14
```
"""
function build_hamiltonian_blocksparse(
    basis::BlockSparseBasis,
    hopping_matrix_input::Union{String, IO},
    coulomb_tensor_input::Union{String, IO}
)
    # Read input files
    hopping_lines = readlines(hopping_matrix_input)
    coulomb_lines = readlines(coulomb_tensor_input)

    # Verify nflavors matches basis
    # Extract max flavor from hopping matrix
    max_flavor_hop = 0
    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            max_flavor_hop = max(max_flavor_hop, i, j)
        end
    end

    # Extract nflavors from coulomb file
    max_flavor_coul = 0
    if !isempty(coulomb_lines)
        # First line is nflavors
        if length(coulomb_lines) >= 1
            max_flavor_coul = parse(Int, coulomb_lines[1]) - 1  # Convert to 0-indexed max
        end
    end

    expected_nflavors = max(max_flavor_hop, max_flavor_coul) + 1

    if basis.nflavors != expected_nflavors
        @warn "Basis nflavors ($(basis.nflavors)) doesn't match input files ($expected_nflavors)"
    end

    # Get annihilation operators
    c = get_annihilators_blocksparse(basis)

    # Build hopping Hamiltonian: H_hop = Σ_{ij} t_{ij} c_i' c_j
    # Start with zero operator (no blocks)
    QN_type = eltype(basis.site_qns)
    H_hop = BlockSparseOperator(
        basis,
        OperatorFlux(zero_qn(QN_type)),
        Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    )

    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1]) + 1  # Convert to 1-indexed
            j = parse(Int, parts[2]) + 1
            val = parse(Float64, parts[3]) + im * parse(Float64, parts[4])

            # Add term: val * c_i' * c_j
            H_hop = H_hop + val * (c[i]' * c[j])
        end
    end

    # Build interaction Hamiltonian: H_int = 0.5 Σ_{ijkl} U_{ijkl} c_i' c_j' c_k c_l
    H_int = BlockSparseOperator(
        basis,
        OperatorFlux(zero_qn(QN_type)),
        Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    )

    if length(coulomb_lines) > 1
        for line in coulomb_lines[2:end]
            parts = split(line)
            i = parse(Int, parts[2]) + 1  # Convert to 1-indexed
            j = parse(Int, parts[3]) + 1
            k = parse(Int, parts[4]) + 1
            l = parse(Int, parts[5]) + 1
            val = parse(Float64, parts[6]) + im * parse(Float64, parts[7])

            # Add term: 0.5 * val * c_i' c_j' c_k c_l
            H_int = H_int + 0.5 * val * (c[i]' * c[j]' * c[k] * c[l])
        end
    end

    # Total Hamiltonian
    H = H_hop + H_int

    return H, c
end

"""
    solve_hamiltonian_blocksparse(H::BlockSparseOperator) -> Tuple{Vector{Float64}, Matrix{ComplexF64}}

Diagonalize a block-sparse Hamiltonian.

This is the block-sparse equivalent of `solve_hamiltonian(H)`.

Arguments:
- `H::BlockSparseOperator`: Block-sparse Hamiltonian

Returns:
- `Vector{Float64}`: Eigenvalues (sorted)
- `Matrix{ComplexF64}`: Eigenvectors (columns)

Note: Currently converts to dense matrix for diagonalization.
Future optimization: Diagonalize each sector independently.

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
H, c = build_hamiltonian_blocksparse(basis, "hopping.txt", "coulomb.txt")

evals, evecs = solve_hamiltonian_blocksparse(H)

# Ground state energy
E0 = evals[1]
```
"""
function solve_hamiltonian_blocksparse(H::BlockSparseOperator)
    # Convert to sparse, then dense
    H_sparse = to_sparse(H)
    H_dense = Matrix(H_sparse)

    # Ensure Hermitian
    sol = eigen(Hermitian(H_dense))

    return sol.values, sol.vectors
end
