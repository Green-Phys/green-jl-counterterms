"""
Hybridization application operations for Propagator representation.

Functions for applying SOE-decomposed hybridization functions K±(τ) to propagators.

Hybridization conserves quantum numbers (flux = 0), so only diagonal blocks are affected.
"""

"""
    apply_hybridization_inplace_impl!(temp_buffer, target, hyb, model_data; is_reverse=false)

Apply hybridization function to target array in-place, updating target with the result.
Uses pre-allocated temp_buffer to avoid allocations.

# Arguments

  - `temp_buffer::Propagator{T}`: Pre-allocated buffer for intermediate results
  - `target::Propagator{T}`: Array to be modified (time-dependent block array)
  - `hyb::HybridizationState`: Hybridization state (NoHyb, KPlus(l), or KMinus(l))
  - `model_data::ModelData{T}`: Model data containing K± values on DLR τ-grid
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
@inline function apply_hybridization_inplace_impl!(
        temp_buffer::Propagator{T}, target::Propagator{T},
        hyb::HybridizationState, model_data::ModelData{T};
        is_reverse::Bool = false) where {T <: Number}
    @boundscheck @assert temp_buffer.basis === target.basis "Buffer and target must use same basis"
    @boundscheck @assert temp_buffer.flux == target.flux "Buffer and target must have same flux"

    index = hyb.index
    if index != 0  # Not NoHyb
        # XOR determines if we should use KPlus: (positive and not reverse) OR (negative and reverse)
        use_kplus = (index > 0) ⊻ is_reverse
        l = abs(index)
        vec = use_kplus ? @view(model_data.hyb.KPlus_on_DLRtau[:, l]) :
              @view(model_data.hyb.KMinus_on_DLRtau[:, l])

        # Apply element-wise multiplication to each block in-place
        # Note: Hybridization is a scalar multiplication, so it applies to all blocks
        # (both diagonal and off-diagonal) in the target array
        for (_, target_block) in target.blocks
            n_tau = size(target_block, 3)  # layout: (dim1, dim2, n_tau)

            # In-place element-wise multiplication: target_block[:, :, k] *= vec[k]
            @inbounds for k in 1:n_tau
                @views target_block[:, :, k] .*= vec[k]
            end
        end
    end
    # For NoHyb, do nothing (target unchanged)
end

#=============================================================================
# Interface Implementations
=============================================================================#

# Hybridization application - interface version
function apply_hybridization_inplace!(
        temp_buffer::Propagator{T}, target::Propagator{T},
        hyb::HybridizationState, model_data::ModelData{T};
        is_reverse::Bool = false) where {T}
    apply_hybridization_inplace_impl!(
        temp_buffer, target, hyb, model_data; is_reverse = is_reverse)
end
