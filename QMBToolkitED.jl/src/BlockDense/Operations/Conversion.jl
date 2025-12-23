"""
Conversion between BlockDenseOperator and Matrix{ComplexF64}.

Functions:
- to_matrix_blockdense: Convert BlockDenseOperator → Matrix{ComplexF64}
- to_vector_blockdense: Extract eigenvalues as a concatenated vector
"""

"""
    to_matrix_blockdense(op::BlockDenseOperator) -> Matrix{ComplexF64}

Convert a BlockDenseOperator to a dense matrix.

Directly constructs a dense matrix by placing blocks at the appropriate global indices.
Each block (target_sector, source_sector) is mapped to the corresponding submatrix
in the global dense matrix.

Arguments:
- `op::BlockDenseOperator`: Block-dense operator

Returns:
- `Matrix{ComplexF64}`: Dense matrix representation

Example:
```julia
basis = make_basis_blockdense([QN_N(1), QN_N(1)])
c1 = get_annihilator_blockdense(basis, 1)

# Convert to dense matrix
c1_mat = to_matrix_blockdense(c1)
@assert size(c1_mat) == (4, 4)  # 2^2 = 4
```
"""
function to_matrix_blockdense(op::BlockDenseOperator)::Matrix{ComplexF64}
    basis = op.basis
    n = basis.total_dim

    # Initialize zero matrix
    result = zeros(ComplexF64, n, n)

    # Compute starting index for each sector
    sector_starts = zeros(Int, length(basis.sectors))
    cumsum_dim = 0
    for (i, sector) in enumerate(basis.sectors)
        sector_starts[i] = cumsum_dim
        cumsum_dim += sector.dim
    end

    # Place each block at the correct position
    for ((tgt_idx, src_idx), block) in op.blocks
        tgt_start = sector_starts[tgt_idx]
        src_start = sector_starts[src_idx]

        tgt_dim = basis.sectors[tgt_idx].dim
        src_dim = basis.sectors[src_idx].dim

        # Copy block to the appropriate submatrix
        result[tgt_start+1:tgt_start+tgt_dim, src_start+1:src_start+src_dim] = block
    end

    return result
end

"""
    to_matrix_blockdense(D::BlockDenseDiagonalOperator) -> Matrix{ComplexF64}

Convert a block-diagonal operator to a full matrix.

Arguments:
- `D::BlockDenseDiagonalOperator`: Diagonal operator

Returns:
- `Matrix{ComplexF64}`: Full diagonal matrix

Example:
```julia
basis = make_basis_blockdense([QN_N(1), QN_N(1)])
sector_diags = Dict(1 => [1.0], 2 => [2.0, 3.0], 3 => [4.0])
D = BlockDenseDiagonalOperator(basis, sector_diags)
mat = to_matrix_blockdense(D)
```
"""
function to_matrix_blockdense(D::BlockDenseDiagonalOperator)::Matrix{ComplexF64}
    basis = D.basis
    n = basis.total_dim

    # Initialize zero matrix
    result = zeros(ComplexF64, n, n)

    # Compute starting index for each sector
    sector_starts = zeros(Int, length(basis.sectors))
    cumsum_dim = 0
    for (i, sector) in enumerate(basis.sectors)
        sector_starts[i] = cumsum_dim
        cumsum_dim += sector.dim
    end

    # Place diagonal elements at correct positions
    for (sector_idx, diag) in D.sector_diags
        sector_start = sector_starts[sector_idx]
        for (i, val) in enumerate(diag)
            result[sector_start + i, sector_start + i] = val
        end
    end

    return result
end

"""
    to_vector_blockdense(D::BlockDenseDiagonalOperator) -> Vector{Float64}

Extract all diagonal values as a single vector, concatenated in sector order.

Arguments:
- `D::BlockDenseDiagonalOperator`: Diagonal operator

Returns:
- `Vector{Float64}`: Concatenated diagonal elements

Example:
```julia
basis = make_basis_blockdense([QN_N(1), QN_N(1)])
sector_diags = Dict(1 => [1.0], 2 => [2.0, 3.0], 3 => [4.0])
D = BlockDenseDiagonalOperator(basis, sector_diags)
vec = to_vector_blockdense(D)  # [1.0, 2.0, 3.0, 4.0]
```
"""
function to_vector_blockdense(D::BlockDenseDiagonalOperator)
    basis = D.basis
    result = Float64[]

    for sector_idx in 1:length(basis.sectors)
        if haskey(D.sector_diags, sector_idx)
            append!(result, D.sector_diags[sector_idx])
        else
            # If sector not in dict, append zeros
            append!(result, zeros(Float64, basis.sectors[sector_idx].dim))
        end
    end

    return result
end
