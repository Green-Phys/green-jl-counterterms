"""
BlockDense reflection operations.

Reflection F(τ) → F(β - τ) is applied independently to each block.
"""

using Lehmann

"""
    do_reflection_blockdense(dlr_grid::DLRGrid, H_dlr::BlockDenseArray{T}) where T<:Number

Perform reflection operation on block-dense array: F(τ) = H(β - τ).

Each block is reflected independently using standard DLR reflection.

# Arguments
- `dlr_grid::DLRGrid`: DLR grid
- `H_dlr::BlockDenseArray{T}`: Block-dense array in DLR representation

# Returns
- `BlockDenseArray{T}`: Reflected array on tau grid
"""
function do_reflection_blockdense(dlr_grid::DLRGrid, H_dlr::BlockDenseArray{T}) where T<:Number
    tau_grid_reflected = dlr_grid.beta .- dlr_grid.tau
    F_tau_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_dlr) in H_dlr.blocks
        # Reflect this block
        F_block_tau = dlr2tau(dlr_grid, block_dlr, tau_grid_reflected)
        F_tau_blocks[(target, source)] = F_block_tau
    end

    return BlockDenseArray{T}(H_dlr.basis, H_dlr.flux, F_tau_blocks)
end

"""
    do_reflection_tau2tau_blockdense(dlr_grid::DLRGrid, H_tau::BlockDenseArray{T}) where T<:Number

Perform reflection operation on tau-space BlockDense data: F(τ) = H(β - τ).
First transforms to DLR, then applies reflection.

# Arguments
- `dlr_grid::DLRGrid`: DLR grid
- `H_tau::BlockDenseArray{T}`: Block-dense array on tau grid

# Returns
- `BlockDenseArray{T}`: Reflected array on tau grid
"""
function do_reflection_tau2tau_blockdense(dlr_grid::DLRGrid, H_tau::BlockDenseArray{T}) where T<:Number
    H_dlr = tau2dlr_blockdense(dlr_grid, H_tau, dlr_grid.tau)
    return do_reflection_blockdense(dlr_grid, H_dlr)
end
