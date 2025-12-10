"""
Fermionic operator construction for dense method.

Constructs annihilation operators as dense matrices in the Fock basis.
Uses sparse internally for efficient construction, then converts to dense.
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

Implementation:
Uses sparse matrix construction internally for efficiency, then converts to dense.
This is faster than building dense matrices element-by-element.
"""
function get_annihilators_dense(nflavors::Int)
    nfock = 1 << nflavors

    # Annihilation operators (will be converted to dense)
    c = Vector{Matrix{ComplexF64}}(undef, nflavors)

    # Sign for fermionic anti-commutation relations
    sign = ones(Int8, nfock)

    for i in 1:nflavors
        bitmask = 1 << (i - 1)

        # Build sparse first (efficient construction)
        num_non_zero = nfock >> 1
        I = Vector{Int}(undef, num_non_zero)
        J = Vector{Int}(undef, num_non_zero)
        V = Vector{Int8}(undef, num_non_zero)
        count = 0

        for alpha in 1:nfock
            fock_state = alpha - 1 # 0-indexed fock state
            if (fock_state & bitmask) != 0
                count += 1
                # State i is occupied, can be annihilated
                I[count] = (fock_state - bitmask) + 1
                J[count] = alpha
                V[count] = sign[alpha]

                # Update sign for the next operator c_{i+1}
                sign[alpha] *= -1
            end
        end
        # Convert to dense for output
        c[i] = Matrix{ComplexF64}(sparse(I, J, V, nfock, nfock))
    end

    return c
end
