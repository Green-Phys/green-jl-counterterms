"""
Conversion between block-dense operators and dense matrices.

Functions:
- to_matrix: Convert Operator{T}/DiagonalOperator → Matrix{T}/Matrix{Float64}
- to_vector: Extract eigenvalues as a concatenated vector
"""

"""
    sector_offsets(basis::Basis) -> Vector{Int}

Compute the starting index (0-based) for each sector in the global matrix.
"""
function sector_offsets(basis::Basis)
    offsets = Vector{Int}(undef, length(basis.sectors))
    cum = 0
    for (i, sector) in enumerate(basis.sectors)
        offsets[i] = cum
        cum += sector.dim
    end
    return offsets
end

"""
    to_matrix(op::Operator{T}) -> Matrix{T}

Convert an Operator to a dense matrix.

Directly constructs a dense matrix by placing blocks at the appropriate global indices.
Each block (target_sector, source_sector) is mapped to the corresponding submatrix
in the global dense matrix.

Arguments:
- `op::Operator{T}`: Block-dense operator

Returns:
- `Matrix{T}`: Dense matrix representation (Float64 for real operators, ComplexF64 for complex)

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
c1 = annihilator(basis, 1)

# Convert to dense matrix
c1_mat = to_matrix(c1)
@assert size(c1_mat) == (4, 4)  # 2^2 = 4
```
"""
function to_matrix(op::Operator{T})::Matrix{T} where {T}
    basis = op.basis
    n = basis.total_dim
    result = zeros(T, n, n)
    offsets = sector_offsets(basis)

    # Place each block at the correct position
    for ((tgt_idx, src_idx), block) in op.blocks
        tgt_start = offsets[tgt_idx]
        src_start = offsets[src_idx]

        tgt_dim = basis.sectors[tgt_idx].dim
        src_dim = basis.sectors[src_idx].dim

        # Copy block to the appropriate submatrix
        result[(tgt_start+1):(tgt_start+tgt_dim), (src_start+1):(src_start+src_dim)] = block
    end

    return result
end

"""
    to_matrix(D::DiagonalOperator) -> Matrix{Float64}

Convert a diagonal operator to a full matrix.

Arguments:
- `D::DiagonalOperator`: Diagonal operator

Returns:
- `Matrix{Float64}`: Full diagonal matrix (eigenvalues are always real)

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
sector_diags = Dict(1 => [1.0], 2 => [2.0, 3.0], 3 => [4.0])
D = DiagonalOperator(basis, QN_N(0), sector_diags)
mat = to_matrix(D)
```
"""
function to_matrix(D::DiagonalOperator)::Matrix{Float64}
    basis = D.basis
    n = basis.total_dim
    result = zeros(Float64, n, n)
    offsets = sector_offsets(basis)

    # Place diagonal elements at correct positions
    for (sector_idx, diag) in D.sector_diags
        sector_start = offsets[sector_idx]
        for (i, val) in enumerate(diag)
            result[sector_start+i, sector_start+i] = val
        end
    end

    return result
end

"""
    to_vector(D::DiagonalOperator) -> Vector{Float64}

Extract all diagonal values as a single vector, concatenated in sector order.

Arguments:
- `D::DiagonalOperator`: Diagonal operator

Returns:
- `Vector{Float64}`: Concatenated diagonal elements

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
sector_diags = Dict(1 => [1.0], 2 => [2.0, 3.0], 3 => [4.0])
D = DiagonalOperator(basis, QN_N(0), sector_diags)
vec = to_vector(D)  # [1.0, 2.0, 3.0, 4.0]
```
"""
function to_vector(D::DiagonalOperator)
    basis = D.basis
    result = zeros(Float64, basis.total_dim)
    offsets = sector_offsets(basis)

    for sector_idx = 1:length(basis.sectors)
        if haskey(D.sector_diags, sector_idx)
            offset = offsets[sector_idx]
            diag = D.sector_diags[sector_idx]
            result[(offset+1):(offset+length(diag))] = diag
        end
    end

    return result
end
