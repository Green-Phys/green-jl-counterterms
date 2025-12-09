"""
Utility functions for block-sparse exact diagonalization.

Includes:
- sum_occupied_sites: Compute total quantum number from occupied sites
- compute_fermion_sign: Local fermionic sign calculation
"""

"""
    sum_occupied_sites(state::Int, site_qns::Vector{QN}) -> QN where QN <: QuantumNumber

Compute the total quantum number of a many-body state by summing the quantum numbers
of all occupied single-particle sites.

Arguments:
- `state::Int`: Bit-pattern integer (0-indexed) representing the Fock state
- `site_qns::Vector{QN}`: Quantum number for each single-particle orbital

Returns:
- Total quantum number obtained by summing site_qns[i+1] for all occupied bits i

The state is interpreted as a bit pattern where bit i (0-indexed) corresponds to
the occupation of orbital i. Julia arrays are 1-indexed, so bit i maps to site_qns[i+1].

Example:
```julia
# 2-site spin system: [↑, ↓]
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]

# State |10⟩ (binary) = 2 (decimal): bit 1 set → orbital 1 occupied (0-indexed)
# This corresponds to site_qns[2] (Julia's 1-indexed array), which is the ↓ spin
state = 2
total_qn = sum_occupied_sites(state, site_qns)  # QN_NSz(1, -1)

# State |11⟩ (binary) = 3 (decimal): both orbitals occupied
state = 3
total_qn = sum_occupied_sites(state, site_qns)  # QN_NSz(2, 0)
```
"""
function sum_occupied_sites(state::Int, site_qns::Vector{QN}) where {QN <: QuantumNumber}
    total = zero_qn(QN)

    for bit in 0:(length(site_qns) - 1)
        # Check if bit is set (orbital occupied)
        if (state >> bit) & 1 == 1
            # Add quantum number of this site (Julia 1-indexed)
            total = combine_qn(total, site_qns[bit + 1])
        end
    end

    return total
end

"""
    compute_fermion_sign(state::Int, flavor::Int) -> Int

Compute the fermionic sign for annihilating orbital `flavor` in `state`.

The sign is (-1)^k where k is the number of occupied orbitals to the right
of `flavor` (i.e., in lower bit positions: bits 0 through flavor-2).

Arguments:
- `state::Int`: Bit-pattern integer (0-indexed) representing the Fock state
- `flavor::Int`: 1-indexed orbital index to annihilate

Returns:
- +1 or -1 depending on the parity of occupied orbitals

This implements the Jordan-Wigner string for fermionic anti-commutation relations.
The convention is that operators are ordered left-to-right in increasing orbital index,
so we count occupied orbitals to the right (lower bit positions).

Example:
```julia
# State |1010⟩ (binary) = 10 (decimal): orbitals 1 and 3 occupied (0-indexed)
state = 10

# Annihilate orbital 2 (1-indexed flavor, bit 1 in 0-indexed)
sign = compute_fermion_sign(state, 2)
# Orbitals below bit 1: bit 0 is NOT set → 0 occupied orbitals
# sign = (-1)^0 = +1

# Annihilate orbital 4 (1-indexed flavor, bit 3 in 0-indexed)
sign = compute_fermion_sign(state, 4)
# Orbitals below bit 3: bits 0,1,2 with only bit 1 set → 1 occupied orbital
# sign = (-1)^1 = -1
```
"""
function compute_fermion_sign(state::Int, flavor::Int)
    # Convert 1-indexed flavor to 0-indexed bit position
    bit_position = flavor - 1

    # Create mask for all bits to the right (lower bit positions)
    # For bit_position=3, mask = 0b0111 = 7 (bits 0,1,2)
    mask = (1 << bit_position) - 1

    # Count occupied orbitals in masked region
    num_occupied = count_ones(state & mask)

    # Fermionic sign: (-1)^num_occupied
    return (-1)^num_occupied
end
