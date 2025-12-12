"""
Hybridization application operations for BlockDense representation.

Functions for applying SOE-decomposed hybridization functions K±(τ) to BlockDense propagators.

Hybridization conserves quantum numbers (flux = 0), so only diagonal blocks are affected.
"""

using QMBToolkitED.BlockDense

"""
    apply_hybridization_inplace_blockdense!(temp_buffer, target, hyb, model_data; is_reverse=false)

Apply hybridization function to BlockDense target array in-place, updating target with the result.
Uses pre-allocated temp_buffer to avoid allocations.

# Arguments
- `temp_buffer::BlockDenseArray{T}`: Pre-allocated buffer for intermediate results
- `target::BlockDenseArray{T}`: Array to be modified (time-dependent block array)
- `hyb::HybridizationState`: Hybridization state (NoHyb, KPlus(l), or KMinus(l))
- `model_data::ModelData{T, BlockDenseFormat}`: Model data containing K± values on DLR τ-grid
- `is_reverse::Bool=false`: When true, swaps KPlus ↔ KMinus (used in reverse convolution)

# Behavior
- For NoHyb (index=0): target unchanged
- For KPlus(l) (index=l>0) and is_reverse=false: multiplies by KPlus_on_DLRtau[:,l]
- For KMinus(l) (index=-l, l>0) and is_reverse=false: multiplies by KMinus_on_DLRtau[:,l]
- When is_reverse=true, the KPlus/KMinus selection is swapped

# Note
Hybridization is applied as element-wise scalar multiplication, so it applies to all blocks
in the target (both diagonal and off-diagonal). The target may have any flux structure
depending on prior operator multiplications in the diagram evaluation.
"""
@inline function apply_hybridization_inplace_blockdense!(temp_buffer::BlockDenseArray{T}, target::BlockDenseArray{T}, hyb::HybridizationState, model_data::ModelData{T, BlockDenseFormat}; is_reverse::Bool=false) where {T<:Number}
    @assert temp_buffer.basis === target.basis "Buffer and target must use same basis"
    @assert temp_buffer.flux.delta_qn == target.flux.delta_qn "Buffer and target must have same flux"

    index = hyb.index
    if index != 0  # Not NoHyb
        # XOR determines if we should use KPlus: (positive and not reverse) OR (negative and reverse)
        use_kplus = (index > 0) ⊻ is_reverse
        l = abs(index)
        vec = use_kplus ? @view(model_data.hyb.KPlus_on_DLRtau[:, l]) :
                          @view(model_data.hyb.KMinus_on_DLRtau[:, l])

        # Apply element-wise multiplication to each block
        # Note: Hybridization is a scalar multiplication, so it applies to all blocks
        # (both diagonal and off-diagonal) in the target array
        for (block_key, target_block) in target.blocks
            temp_block = temp_buffer.blocks[block_key]
            n_tau = size(target_block, 1)

            # Element-wise multiplication: temp_block[k, :, :] = vec[k] * target_block[k, :, :]
            @inbounds for k in 1:n_tau
                @views temp_block[k, :, :] .= vec[k] .* target_block[k, :, :]
            end

            # Copy back to target
            copyto!(target_block, temp_block)
        end
    end
    # For NoHyb, do nothing (target unchanged)
end

"""
    apply_hybridization_blockdense(hyb, result, model_data; is_reverse=false)

Allocating version: creates a new BlockDenseArray for the result.

# Arguments
- `hyb::HybridizationState`: Hybridization state
- `result::BlockDenseArray{T}`: Input array
- `model_data::ModelData{T, BlockDenseFormat}`: Model data containing K± values
- `is_reverse::Bool=false`: When true, swaps KPlus ↔ KMinus

# Returns
- New BlockDenseArray with hybridization applied, or original array if NoHyb
"""
@inline function apply_hybridization_blockdense(hyb::HybridizationState, result::BlockDenseArray{T}, model_data::ModelData{T, BlockDenseFormat}; is_reverse::Bool=false) where {T<:Number}
    index = hyb.index
    if index != 0  # Not NoHyb
        # XOR determines if we should use KPlus: (positive and not reverse) OR (negative and reverse)
        use_kplus = (index > 0) ⊻ is_reverse
        l = abs(index)
        vec = use_kplus ? @view(model_data.hyb.KPlus_on_DLRtau[:, l]) :
                          @view(model_data.hyb.KMinus_on_DLRtau[:, l])

        # Create new blocks with hybridization applied
        # Note: Hybridization is a scalar multiplication, so it applies to all blocks
        new_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

        for (block_key, result_block) in result.blocks
            n_tau = size(result_block, 1)
            dim_target = size(result_block, 2)
            dim_source = size(result_block, 3)
            new_block = zeros(T, n_tau, dim_target, dim_source)

            # Element-wise multiplication: new_block[k, :, :] = vec[k] * result_block[k, :, :]
            @inbounds for k in 1:n_tau
                @views new_block[k, :, :] .= vec[k] .* result_block[k, :, :]
            end

            new_blocks[block_key] = new_block
        end

        return BlockDenseArray{T}(result.basis, result.flux, new_blocks)
    else
        return result
    end
end
