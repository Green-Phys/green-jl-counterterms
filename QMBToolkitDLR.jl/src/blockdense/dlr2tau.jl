"""
BlockDense DLR transformations: DLR → tau.

Transforms each block independently, preserving block structure.
"""

using Lehmann

"""
    dlr2tau_blockdense(dlr_grid::DLRGrid, F_dlr::BlockDenseArray{T}, tau_grid::Vector{Float64}) where T<:Number

Transform block-dense array from DLR representation to tau space.

Each block is transformed independently using standard dlr2tau.

# Arguments
- `dlr_grid::DLRGrid`: DLR grid specification
- `F_dlr::BlockDenseArray{T}`: Block-dense array in DLR representation
- `tau_grid::Vector{Float64}`: Target tau grid points

# Returns
- `BlockDenseArray{T}`: Block-dense array on tau grid
"""
function dlr2tau_blockdense(dlr_grid::DLRGrid, F_dlr::BlockDenseArray{T}, tau_grid::Vector{Float64}) where T<:Number
    F_tau_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_dlr) in F_dlr.blocks
        # Transform this block: (r, n_row, n_col) → (n_tau, n_row, n_col)
        block_tau = dlr2tau(dlr_grid, block_dlr, tau_grid)
        F_tau_blocks[(target, source)] = block_tau
    end

    return BlockDenseArray{T}(F_dlr.basis, F_dlr.flux, F_tau_blocks)
end