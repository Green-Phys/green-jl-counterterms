"""
Hamiltonian construction for dense method.

Builds Hamiltonians from hopping and Coulomb interaction input files.
"""

"""
    build_hamiltonian_dense(hopping_input, coulomb_input) -> (DenseOperator, Vector{DenseOperator})

Construct a Hamiltonian from hopping and Coulomb interaction input files.

The Hamiltonian has the form:
    H = Σ_{ij} t_{ij} c_i† c_j + 0.5 Σ_{ijkl} U_{ijkl} c_i† c_j† c_k c_l

Arguments:
- `hopping_input::Union{String, IO}`: Path to hopping matrix file or IO
  Format: Each line is "i j Re(t_ij) Im(t_ij)" (0-indexed)
- `coulomb_input::Union{String, IO}`: Path to Coulomb tensor file or IO
  Format: First line is nflavors, then "# i j k l Re(U_ijkl) Im(U_ijkl)" (0-indexed)

Returns:
- `DenseOperator`: The total Hamiltonian H
- `Vector{DenseOperator}`: Annihilation operators used to build H

Example:
```julia
H, c = build_hamiltonian_dense("hopping.txt", "coulomb.txt")
```

Implementation:
Operators are DenseOperators. Hamiltonian construction uses operator algebra.
"""
function build_hamiltonian_dense(
    hopping_matrix_input::Union{String, IO},
    coulomb_tensor_input::Union{String, IO}
)
    hopping_lines = readlines(hopping_matrix_input)
    coulomb_lines = readlines(coulomb_tensor_input)

    # Read hopping matrix to determine nflavors
    max_flavor = 0
    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            max_flavor = max(max_flavor, i, j)
        end
    end
    nflavors = max_flavor + 1

    # Create basis and get annihilation operators
    basis = make_basis_dense(nflavors)
    annh = get_annihilators_dense(basis)

    # Build hopping Hamiltonian: H_hop = Σ_{ij} t_{ij} c_i† c_j
    H_hop = DenseOperator(basis, zeros(ComplexF64, basis.dim, basis.dim))
    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1]) + 1  # Convert to 1-indexed
            j = parse(Int, parts[2]) + 1
            val = parse(Float64, parts[3]) + im * parse(Float64, parts[4])
            H_hop = H_hop + val * (annh[i]' * annh[j])
        end
    end

    # Build interaction Hamiltonian: H_int = 0.5 Σ_{ijkl} U_{ijkl} c_i† c_j† c_k c_l
    H_int = DenseOperator(basis, zeros(ComplexF64, basis.dim, basis.dim))
    if length(coulomb_lines) > 1
        for line in coulomb_lines[2:end]
            parts = split(line)
            i = parse(Int, parts[2]) + 1  # Convert to 1-indexed
            j = parse(Int, parts[3]) + 1
            k = parse(Int, parts[4]) + 1
            l = parse(Int, parts[5]) + 1
            val = parse(Float64, parts[6]) + im * parse(Float64, parts[7])
            H_int = H_int + 0.5 * val * (annh[i]' * annh[j]' * annh[k] * annh[l])
        end
    end

    # Total Hamiltonian
    H = H_hop + H_int
    return H, annh
end
