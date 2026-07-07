"""
Hamiltonian construction for quantum many-body systems.

Provides functions to build Hamiltonians from hopping matrix and Coulomb tensor inputs.
"""

"""
    build_hamiltonian(
        basis::Basis,
        hopping_matrix_input::Union{String, IO},
        coulomb_tensor_input::Union{String, IO}
    ) -> Tuple{Operator, Vector{Operator}}

Construct a Hamiltonian from hopping and Coulomb interaction inputs.

The Hamiltonian has the form:
    H = Σ_{ij} t_{ij} c_i' c_j + 0.5 Σ_{ijkl} U_{ijkl} c_i' c_k' c_l c_j

This uses chemist (Mulliken) notation where U_{ijkl} corresponds to the two-electron
integral (ij|kl), with index order [cre1, ann2, cre2, ann1].
The 0.5 prefactor accounts for double-counting in the sum over all indices.

Arguments:
- `basis::Basis`: The many-body basis (must match nflavors in input files)
- `hopping_matrix_input`: Path to file or IO containing hopping matrix
  Format: Each line is "i j Re(t_ij) Im(t_ij)" (0-indexed)
- `coulomb_tensor_input`: Path to file or IO containing Coulomb tensor
  Format: First line is count, then "# i j k l Re(U_ijkl) Im(U_ijkl)" (0-indexed)

Returns:
- `Operator`: The total Hamiltonian H
- `Vector{Operator}`: Annihilation operators used to build H

Note: The Hamiltonian should have flux = 0 (number conserving).

Example:
```julia
flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = Basis(flavor_qns)

H, c = build_hamiltonian(basis, "hopping.txt", "coulomb.txt")

# H should be Hermitian
H_mat = to_matrix(H)
@assert H_mat ≈ H_mat' atol=1e-14

# H should have zero flux (number conserving)
@assert H.flux == QN_N(0)
```
"""
function build_hamiltonian(
    basis::Basis,
    hopping_matrix_input::Union{String,IO},
    coulomb_tensor_input::Union{String,IO},
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

    # Get annihilation operators (still needed for return value)
    c = annihilators(basis)
    QN_type = typeof(basis.flavor_qns[1])

    # Parse hopping terms: (i, j, t_ij) with 1-indexed flavors
    # Track if all terms are real
    all_real = true
    hopping_terms = Tuple{Int,Int,ComplexF64}[]
    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1]) + 1  # Convert to 1-indexed
            j = parse(Int, parts[2]) + 1
            re_val = parse(Float64, parts[3])
            im_val = parse(Float64, parts[4])
            if im_val != 0.0
                all_real = false
            end
            val = re_val + im * im_val
            if !iszero(val)
                push!(hopping_terms, (i, j, val))
            end
        end
    end

    # Parse interaction terms: (i, j, k, l, U_ijkl) with 1-indexed flavors
    # Chemist notation: U[i,j,k,l] -> 0.5 * c†_i c†_k c_l c_j
    interaction_terms = Tuple{Int,Int,Int,Int,ComplexF64}[]
    if length(coulomb_lines) > 1
        for line in coulomb_lines[2:end]
            parts = split(line)
            fl_i = parse(Int, parts[2]) + 1
            fl_j = parse(Int, parts[3]) + 1
            fl_k = parse(Int, parts[4]) + 1
            fl_l = parse(Int, parts[5]) + 1
            re_val = parse(Float64, parts[6])
            im_val = parse(Float64, parts[7])
            if im_val != 0.0
                all_real = false
            end
            val = re_val + im * im_val
            if !iszero(val)
                push!(interaction_terms, (fl_i, fl_j, fl_k, fl_l, val))
            end
        end
    end

    # Build Hamiltonian blocks directly by iterating over Fock states
    # The Hamiltonian has flux = 0 (number conserving), so only diagonal blocks
    # Use Float64 if all terms are real, ComplexF64 otherwise
    T = all_real ? Float64 : ComplexF64
    blocks = Dict{Tuple{Int,Int},Matrix{T}}()

    for (sector_idx, sector) in enumerate(basis.sectors)
        dim = sector.dim
        block = zeros(T, dim, dim)

        # Iterate over all source states in this sector
        for (src_local_idx, src_state) in enumerate(sector.states)

            # Apply hopping terms: t_ij c†_i c_j |src_state⟩
            for (i, j, t_ij) in hopping_terms
                # Step 1: Apply c_j (annihilate flavor j)
                bit_j = 1 << (j - 1)
                if (src_state & bit_j) == 0
                    continue  # Flavor j not occupied, c_j|s⟩ = 0
                end
                sign1 = fermion_sign(src_state, j)
                state1 = src_state ⊻ bit_j  # Flip bit j off

                # Step 2: Apply c†_i (create flavor i)
                bit_i = 1 << (i - 1)
                if (state1 & bit_i) != 0
                    continue  # Flavor i already occupied, c†_i|s1⟩ = 0
                end
                sign2 = fermion_sign(state1, i)
                tgt_state = state1 ⊻ bit_i  # Flip bit i on

                # Look up target state index (must be in same sector since H is number-conserving)
                tgt_local_idx = sector.state_to_index[tgt_state]

                # Accumulate matrix element (use real part when T is Float64)
                coeff = T == Float64 ? real(t_ij) : t_ij
                block[tgt_local_idx, src_local_idx] += coeff * sign1 * sign2
            end

            # Apply interaction terms: 0.5 * U_ijkl c†_i c†_k c_l c_j |src_state⟩
            # Operator order (right to left): c_j, c_l, c†_k, c†_i
            for (i, j, k, l, U_ijkl) in interaction_terms
                state = src_state
                total_sign = 1

                # Step 1: Apply c_j (annihilate flavor j)
                bit_j = 1 << (j - 1)
                if (state & bit_j) == 0
                    continue  # Flavor j not occupied
                end
                total_sign *= fermion_sign(state, j)
                state = state ⊻ bit_j

                # Step 2: Apply c_l (annihilate flavor l)
                bit_l = 1 << (l - 1)
                if (state & bit_l) == 0
                    continue  # Flavor l not occupied
                end
                total_sign *= fermion_sign(state, l)
                state = state ⊻ bit_l

                # Step 3: Apply c†_k (create flavor k)
                bit_k = 1 << (k - 1)
                if (state & bit_k) != 0
                    continue  # Flavor k already occupied
                end
                total_sign *= fermion_sign(state, k)
                state = state ⊻ bit_k

                # Step 4: Apply c†_i (create flavor i)
                bit_i = 1 << (i - 1)
                if (state & bit_i) != 0
                    continue  # Flavor i already occupied
                end
                total_sign *= fermion_sign(state, i)
                tgt_state = state ⊻ bit_i

                # Look up target state index
                tgt_local_idx = sector.state_to_index[tgt_state]

                # Accumulate matrix element (with 0.5 prefactor, use real part when T is Float64)
                coeff = T == Float64 ? real(U_ijkl) : U_ijkl
                block[tgt_local_idx, src_local_idx] += 0.5 * coeff * total_sign
            end
        end

        # Store non-zero blocks (diagonal blocks only for H with flux=0)
        if !iszero(block)
            blocks[(sector_idx, sector_idx)] = block
        end
    end

    H = Operator{T}(basis, zero(QN_type), blocks)

    return H, c
end


# =============================================================================
# Original operator-algebra implementation (commented out for reference)
# =============================================================================
#=
function build_hamiltonian_operator_algebra(
    basis::Basis,
    hopping_matrix_input::Union{String,IO},
    coulomb_tensor_input::Union{String,IO},
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
    c = annihilators(basis)
    QN_type = typeof(basis.flavor_qns[1])

    # Build hopping Hamiltonian: H_hop = Σ_{ij} t_{ij} c_i' c_j
    # Collect all terms, then sum once (avoids repeated intermediate allocations)
    hop_terms = Operator[]

    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1]) + 1  # Convert to 1-indexed
            j = parse(Int, parts[2]) + 1
            val = parse(Float64, parts[3]) + im * parse(Float64, parts[4])

            # Skip zero hopping terms to avoid flux mismatch
            if iszero(val)
                continue
            end

            push!(hop_terms, val * (c[i]' * c[j]))
        end
    end

    H_hop = if isempty(hop_terms)
        Operator(basis, zero(QN_type), Dict{Tuple{Int,Int},Matrix{ComplexF64}}())
    else
        sum(hop_terms)
    end

    # Build interaction Hamiltonian: H_int = 0.5 Σ_{ijkl} U_{ijkl} c_i' c_k' c_l c_j
    # Uses chemist (Mulliken) notation: U[i,j,k,l] -> 0.5 * c†_i c†_k c_l c_j
    # Collect all terms, then sum once
    int_terms = Operator[]

    if length(coulomb_lines) > 1
        for line in coulomb_lines[2:end]
            parts = split(line)
            fl_i = parse(Int, parts[2]) + 1
            fl_j = parse(Int, parts[3]) + 1
            fl_k = parse(Int, parts[4]) + 1
            fl_l = parse(Int, parts[5]) + 1
            val = parse(Float64, parts[6]) + im * parse(Float64, parts[7])

            # Skip zero interaction terms to avoid flux mismatch
            if iszero(val)
                continue
            end

            # Chemist notation: U[i,j,k,l] -> 0.5 * c†_i c†_k c_l c_j
            push!(int_terms, 0.5 * val * (c[fl_i]' * c[fl_k]' * c[fl_l] * c[fl_j]))
        end
    end

    H_int = if isempty(int_terms)
        Operator(basis, zero(QN_type), Dict{Tuple{Int,Int},Matrix{ComplexF64}}())
    else
        sum(int_terms)
    end

    # Total Hamiltonian
    H = H_hop + H_int

    return H, c
end
=#
