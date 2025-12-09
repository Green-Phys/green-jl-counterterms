"""
Conversion between block-sparse and sparse matrix formats.

Provides:
- to_sparse: Convert BlockSparseOperator to standard sparse matrix
- from_sparse: Convert sparse matrix to BlockSparseOperator (with known flux)
"""

"""
    to_sparse(op::BlockSparseOperator) -> SparseMatrixCSC{ComplexF64, Int}

Convert a block-sparse operator to a standard sparse matrix.

The sparse matrix has dimensions (total_dim × total_dim) where total_dim = 2^nflavors.
Each block is mapped to the appropriate global indices in the sparse matrix.

Arguments:
- `op::BlockSparseOperator`: Block-sparse operator

Returns:
- `SparseMatrixCSC{ComplexF64, Int}`: Sparse matrix representation

The mapping works as follows:
- Each sector has a starting global index (cumulative sum of previous sector dimensions)
- Block[i,j] in sector block (tgt_sector, src_sector) maps to global index:
  - global_row = tgt_sector_start + i
  - global_col = src_sector_start + j

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blocksparse(basis, 1)

# Convert to sparse for validation
c1_sparse = to_sparse(c1)
@assert size(c1_sparse) == (4, 4)  # 2^2 = 4
```
"""
function to_sparse(op::BlockSparseOperator)
    basis = op.basis
    n = basis.total_dim

    # Compute starting index for each sector
    # Sectors are ordered, so we can compute cumulative sum
    sector_starts = zeros(Int, length(basis.sectors))
    cumsum_dim = 0
    for (i, sector) in enumerate(basis.sectors)
        sector_starts[i] = cumsum_dim
        cumsum_dim += sector.dim
    end

    # Collect all non-zero entries
    I = Int[]
    J = Int[]
    V = ComplexF64[]

    for ((tgt_idx, src_idx), block) in op.blocks
        tgt_start = sector_starts[tgt_idx]
        src_start = sector_starts[src_idx]

        # Map each block element to global indices
        for j in 1:size(block, 2)  # source states (columns)
            for i in 1:size(block, 1)  # target states (rows)
                val = block[i, j]
                if !iszero(val)
                    push!(I, tgt_start + i)
                    push!(J, src_start + j)
                    push!(V, val)
                end
            end
        end
    end

    # Create sparse matrix
    return sparse(I, J, V, n, n)
end

"""
    from_sparse(matrix::AbstractMatrix, basis::BlockSparseBasis, flux::OperatorFlux) -> BlockSparseOperator

Convert a sparse matrix to a block-sparse operator.

This function partitions the input matrix into blocks according to the basis sectors
and the specified flux. Only blocks connecting compatible sectors are extracted.

Arguments:
- `matrix::AbstractMatrix`: Input matrix (sparse or dense)
- `basis::BlockSparseBasis`: Basis defining the block structure
- `flux::OperatorFlux`: Quantum number flux of the operator

Returns:
- `BlockSparseOperator`: Block-sparse representation

Note: This is primarily for testing and validation. The matrix must respect
the sector structure, otherwise blocks may be zero or incorrect.

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator_blocksparse(basis, 1)

# Convert to sparse and back
c1_sparse = to_sparse(c1)
c1_reconstructed = from_sparse(c1_sparse, basis, c1.flux)

# Should match original
@assert all(c1.blocks[key] ≈ c1_reconstructed.blocks[key] for key in keys(c1.blocks))
```
"""
function from_sparse(matrix::AbstractMatrix, basis::BlockSparseBasis, flux::OperatorFlux)
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

        block = Matrix{ComplexF64}(matrix[tgt_range, src_range])

        # Store if non-zero
        if !iszero(block)
            blocks[(tgt_idx, src_idx)] = block
        end
    end

    return BlockSparseOperator(basis, flux, blocks)
end
