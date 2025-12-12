"""
BlockDense DLR transformations: tau → DLR.

Transforms each block independently, preserving block structure.
"""

using Lehmann
using QMBToolkitED.BlockDense

"""
    BlockDenseArray{T}

Time-dependent block-dense array for DLR operations.

Each block is stored as a 3D array: (time_points, row_dim, col_dim)
where time_points is either tau grid or DLR grid.

Fields:
- `basis::BlockDenseBasis`: Basis defining the block structure
- `flux::OperatorFlux`: Quantum number change (flux) of the operator
- `blocks::Dict{Tuple{Int,Int}, Array{T,3}}`: Blocks indexed by (target_sector, source_sector)
"""
struct BlockDenseArray{T<:Number}
    basis::BlockDenseBasis
    flux::OperatorFlux
    blocks::Dict{Tuple{Int,Int}, Array{T,3}}
end

"""
    tau2dlr_blockdense(dlr_grid::DLRGrid, F_tau::BlockDenseArray{T}, tau_grid::Vector{Float64}) where T<:Number

Transform block-dense array from tau space to DLR representation.

Each block is transformed independently using standard tau2dlr.

# Arguments
- `dlr_grid::DLRGrid`: DLR grid specification
- `F_tau::BlockDenseArray{T}`: Block-dense array on tau grid
- `tau_grid::Vector{Float64}`: Tau grid points

# Returns
- `BlockDenseArray{T}`: Block-dense array in DLR representation
"""
function tau2dlr_blockdense(dlr_grid::DLRGrid, F_tau::BlockDenseArray{T}, tau_grid::Vector{Float64}) where T<:Number
    F_dlr_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_tau) in F_tau.blocks
        # Transform this block: (n_tau, n_row, n_col) → (r, n_row, n_col)
        block_dlr = tau2dlr(dlr_grid, block_tau, tau_grid)
        F_dlr_blocks[(target, source)] = block_dlr
    end

    return BlockDenseArray{T}(F_tau.basis, F_tau.flux, F_dlr_blocks)
end