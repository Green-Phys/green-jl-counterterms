"""
High-level constructors for block-dense operators.

Provides convenient functions that return collections of operators.
"""

"""
    get_annihilators_blockdense(basis::BlockDenseBasis) -> Vector{BlockDenseOperator}

Generate fermionic annihilation operators for a given block-dense basis.

This is the block-dense equivalent of `get_annihilators(nflavors)`.

Arguments:
- `basis::BlockDenseBasis`: The many-body basis with sector structure

Returns:
- `Vector{BlockDenseOperator}`: Array of annihilation operators c[i] for i=1:nflavors

Example:
```julia
# Create basis for 3-orbital system with N conservation
site_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = make_basis(site_qns)

# Get annihilation operators
c = get_annihilators_blockdense(basis)

# c[i] is the annihilation operator for flavor i
```

Comparison with sparse version:
```julia
# Sparse version
c_sparse = get_annihilators(3)

# Block-sparse version
basis = make_basis([QN_N(1), QN_N(1), QN_N(1)])
c_block = get_annihilators_blockdense(basis)

# Should match when converted to sparse
for i in 1:3
    @assert to_sparse(c_block[i]) ≈ c_sparse[i]
end
```
"""
function get_annihilators_blockdense(basis::BlockDenseBasis)
    return [get_annihilator_blockdense(basis, i) for i in 1:basis.nflavors]
end
