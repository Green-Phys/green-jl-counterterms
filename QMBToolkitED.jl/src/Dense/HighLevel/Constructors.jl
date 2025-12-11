"""
Fermionic operator construction for dense method.

Constructs annihilation operators as dense matrices in the Fock basis.
"""

"""
    get_annihilators_dense(nflavors::Int) -> Vector{Matrix{ComplexF64}}

Generate fermionic annihilation operators for a Fock space with `nflavors`.

The operators are represented as dense matrices in the occupation number basis.
Fock states are ordered by their binary representation: state |100⟩ = 4.

Arguments:
- `nflavors::Int`: Number of orbitals/flavors

Returns:
- `Vector{Matrix{ComplexF64}}`: Array of annihilation operators c[i]

Example:
```julia
c = get_annihilators_dense(3)
# c[i] is the annihilation operator for flavor i
# Returns dense matrices (2^3 × 2^3)
```
"""
function get_annihilators_dense(nflavors::Int)
    nfock = 1 << nflavors

    # Annihilation operators
    c = Vector{Matrix{ComplexF64}}(undef, nflavors)

    # Sign for fermionic anti-commutation relations
    sign = ones(Int8, nfock)

    for i in 1:nflavors
        bitmask = 1 << (i - 1)

        # Build dense matrix directly
        c[i] = zeros(ComplexF64, nfock, nfock)

        for alpha in 1:nfock
            fock_state = alpha - 1 # 0-indexed fock state
            if (fock_state & bitmask) != 0
                # State i is occupied, can be annihilated
                row = (fock_state - bitmask) + 1
                c[i][row, alpha] = sign[alpha]

                # Update sign for the next operator c_{i+1}
                sign[alpha] *= -1
            end
        end
    end

    return c
end
