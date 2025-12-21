"""
Fermionic operator construction for dense method.

Constructs annihilation operators as DenseOperators in the Fock basis.
"""

"""
    get_annihilators_dense(basis::DenseBasis) -> Vector{DenseOperator}

Generate fermionic annihilation operators for a Fock space.

The operators are represented as DenseOperators in the occupation number basis.
Fock states are ordered by their binary representation: state |100⟩ = 4.

Arguments:
- `basis::DenseBasis`: The dense basis

Returns:
- `Vector{DenseOperator}`: Array of annihilation operators c[i]

Example:
```julia
basis = make_basis_dense(3)
c = get_annihilators_dense(basis)
# c[i] is the annihilation operator for flavor i
```
"""
function get_annihilators_dense(basis::DenseBasis)
    nflavors = basis.nflavors
    nfock = basis.dim

    # Annihilation operators
    c = Vector{DenseOperator}(undef, nflavors)

    # Sign for fermionic anti-commutation relations
    sign = ones(Int8, nfock)

    for i in 1:nflavors
        bitmask = 1 << (i - 1)

        # Build dense matrix directly
        matrix = zeros(ComplexF64, nfock, nfock)

        for alpha in 1:nfock
            fock_state = alpha - 1 # 0-indexed fock state
            if (fock_state & bitmask) != 0
                # State i is occupied, can be annihilated
                row = (fock_state - bitmask) + 1
                matrix[row, alpha] = sign[alpha]

                # Update sign for the next operator c_{i+1}
                sign[alpha] *= -1
            end
        end

        c[i] = DenseOperator(basis, matrix)
    end

    return c
end

"""
    get_annihilators_dense(nflavors::Int) -> Vector{DenseOperator}

Generate fermionic annihilation operators for a Fock space with `nflavors`.

This is a convenience method that creates a DenseBasis internally.

Arguments:
- `nflavors::Int`: Number of orbitals/flavors

Returns:
- `Vector{DenseOperator}`: Array of annihilation operators c[i]

Example:
```julia
c = get_annihilators_dense(3)
# c[i] is the annihilation operator for flavor i
# Returns DenseOperators (2^3 × 2^3)
```
"""
function get_annihilators_dense(nflavors::Int)
    basis = make_basis_dense(nflavors)
    return get_annihilators_dense(basis)
end
