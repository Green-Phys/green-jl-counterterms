"""
High-level Hamiltonian construction for block-dense operators.

Provides functions to build Hamiltonians from input files.
"""

"""
    build_hamiltonian_blockdense(
        basis::BlockDenseBasis,
        hopping_matrix_input::Union{String, IO},
        coulomb_tensor_input::Union{String, IO}
    ) -> Tuple{BlockDenseOperator, Vector{BlockDenseOperator}}

Construct a block-dense Hamiltonian from hopping and Coulomb interaction inputs.

This is the block-dense equivalent of `build_hamiltonian(hopping, coulomb)`.

The Hamiltonian has the form:
    H = Σ_{ij} t_{ij} c_i' c_j + 0.5 Σ_{ijkl} U_{ijkl} c_i' c_j' c_k c_l

Arguments:
- `basis::BlockDenseBasis`: The many-body basis (must match nflavors in input files)
- `hopping_matrix_input`: Path to file or IO containing hopping matrix
  Format: Each line is "i j Re(t_ij) Im(t_ij)" (0-indexed)
- `coulomb_tensor_input`: Path to file or IO containing Coulomb tensor
  Format: First line is nflavors, then "# i j k l Re(U_ijkl) Im(U_ijkl)" (0-indexed)

Returns:
- `BlockDenseOperator`: The total Hamiltonian H
- `Vector{BlockDenseOperator}`: Annihilation operators used to build H

Note: The Hamiltonian should have flux = 0 (number conserving).

Example:
```julia
site_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = make_basis_blockdense(site_qns)

H, c = build_hamiltonian_blockdense(basis, "hopping.txt", "coulomb.txt")

# H should be Hermitian
H_mat = to_matrix(H)
@assert H_mat ≈ H_mat' atol=1e-14

# H should have zero flux (number conserving)
@assert H.flux.delta_qn == QN_N(0)
```

Comparison with Dense version:
```julia
# Dense version
H_dense, c_dense = build_hamiltonian("hopping.txt", "coulomb.txt")

# BlockDense version
basis = make_basis_blockdense([QN_N(1) for _ in 1:nflavors])
H_block, c_block = build_hamiltonian_blockdense(basis, "hopping.txt", "coulomb.txt")

# Hamiltonians should match
@assert to_matrix(H_block) ≈ to_matrix(H_dense) atol=1e-14
```
"""
function build_hamiltonian_blockdense(
    basis::BlockDenseBasis,
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
        # First line is the number of non-zero flavors, skip it
        for line in coulomb_lines[2:end]
            parts = split(line)
            fl_i = parse(Int, parts[2])
            fl_j = parse(Int, parts[3])
            fl_k = parse(Int, parts[4])
            fl_l = parse(Int, parts[5])
            max_flavor_coul = max(max_flavor_coul, fl_i, fl_j, fl_k, fl_l)
        end
    end

    expected_nflavors = max(max_flavor_hop, max_flavor_coul) + 1

    if basis.nflavors != expected_nflavors
        @warn "Basis nflavors ($(basis.nflavors)) doesn't match input files ($expected_nflavors)"
    end

    # Get annihilation operators
    c = get_annihilators_blockdense(basis)
    QN_type = typeof(basis.site_qns[1])

    # Build hopping Hamiltonian: H_hop = Σ_{ij} t_{ij} c_i' c_j
    # Start with zero operator (no blocks)
    H_hop = BlockDenseOperator(
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

            H_hop = H_hop + val * (c[i]' * c[j])
        end
    end

    # Build interaction Hamiltonian: H_int = 0.5 Σ_{ijkl} U_{ijkl} c_i' c_j' c_k c_l
    H_int = BlockDenseOperator(
        basis,
        OperatorFlux(zero_qn(QN_type)),
        Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    )

    if length(coulomb_lines) > 1
        for line in coulomb_lines[2:end]
            parts = split(line)
            fl_i = parse(Int, parts[2]) + 1
            fl_j = parse(Int, parts[3]) + 1
            fl_k = parse(Int, parts[4]) + 1
            fl_l = parse(Int, parts[5]) + 1
            val = parse(Float64, parts[6]) + im * parse(Float64, parts[7])

            H_int = H_int + 0.5 * val * (c[fl_i]' * c[fl_j]' * c[fl_k] * c[fl_l])
        end
    end

    # Total Hamiltonian
    H = H_hop + H_int

    return H, c
end
