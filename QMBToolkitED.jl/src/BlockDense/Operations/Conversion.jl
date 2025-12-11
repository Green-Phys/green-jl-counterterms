"""
Conversion between BlockDenseOperator and Matrix{ComplexF64}.

Functions:
- to_dense: Convert BlockDenseOperator → Matrix{ComplexF64}
- from_dense: Convert Matrix{ComplexF64} → BlockDenseOperator (with known flux)
"""

"""
    to_dense(op::BlockDenseOperator) -> Matrix{ComplexF64}

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
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blockdense(basis, 1)

# Convert to dense matrix
c1_dense = to_dense(c1)
@assert size(c1_dense) == (4, 4)  # 2^2 = 4
```
"""
function to_dense(op::BlockDenseOperator)
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
    from_dense(matrix::Matrix{ComplexF64}, basis::BlockDenseBasis, flux::OperatorFlux) -> BlockDenseOperator

Convert a dense matrix to a BlockDenseOperator.

Extracts blocks from the dense matrix according to the basis sector structure
and the specified flux. Only blocks connecting compatible sectors are extracted.

Arguments:
- `matrix::Matrix{ComplexF64}`: Input dense matrix
- `basis::BlockDenseBasis`: Basis defining the block structure
- `flux::OperatorFlux`: Quantum number flux of the operator

Returns:
- `BlockDenseOperator`: Block-dense representation

Note: The matrix must respect the sector structure implied by the flux,
otherwise extracted blocks may be zero or incorrect.

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blockdense(basis, 1)

# Convert to dense and back
c1_dense = to_dense(c1)
c1_reconstructed = from_dense(c1_dense, basis, c1.flux)

# Should match original
@assert all(c1.blocks[key] ≈ c1_reconstructed.blocks[key] for key in keys(c1.blocks))
```
"""
function from_dense(matrix::Matrix{ComplexF64}, basis::BlockDenseBasis, flux::OperatorFlux)
    @assert size(matrix) == (basis.total_dim, basis.total_dim) "Matrix size must match basis dimension"

    # Compute sector starting indices
    sector_starts = zeros(Int, length(basis.sectors))
    cumsum_dim = 0
    for (i, sector) in enumerate(basis.sectors)
        sector_starts[i] = cumsum_dim
        cumsum_dim += sector.dim
    end

    # Extract blocks for compatible sector pairs
    blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    for (src_idx, src_sector) in enumerate(basis.sectors)
        # Compute target quantum number
        target_qn = combine_qn(src_sector.qn, flux.delta_qn)

        # Check if target sector exists
        if !haskey(basis.qn_to_sector, target_qn)
            continue
        end

        tgt_idx = basis.qn_to_sector[target_qn]
        tgt_sector = basis.sectors[tgt_idx]

        # Extract block from global matrix
        tgt_start = sector_starts[tgt_idx]
        src_start = sector_starts[src_idx]

        tgt_range = (tgt_start + 1):(tgt_start + tgt_sector.dim)
        src_range = (src_start + 1):(src_start + src_sector.dim)

        block = matrix[tgt_range, src_range]

        # Store if non-zero
        if !iszero(block)
            blocks[(tgt_idx, src_idx)] = block
        end
    end

    return BlockDenseOperator(basis, flux, blocks)
end
