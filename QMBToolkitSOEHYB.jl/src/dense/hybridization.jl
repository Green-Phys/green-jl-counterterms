"""
Hybridization application operations for dense representation.

Functions for applying SOE-decomposed hybridization functions K±(τ) to propagators.
"""

"""
    apply_hybridization_inplace!(temp_buffer, target, hyb, model_data; is_reverse=false)

Apply hybridization function to target array in-place, updating target with the result.
Uses pre-allocated temp_buffer to avoid allocations.

# Arguments
- `temp_buffer::Array{T,3}`: Pre-allocated buffer for intermediate results
- `target::Array{T,3}`: Array to be modified (τ, orbital1, orbital2)
- `hyb::HybridizationState`: Hybridization state (NoHyb, KPlus(l), or KMinus(l))
- `model_data::ModelData{T}`: Model data containing K± values on DLR τ-grid
- `is_reverse::Bool=false`: When true, swaps KPlus ↔ KMinus (used in reverse convolution)

# Behavior
- For NoHyb (index=0): target unchanged
- For KPlus(l) (index=l>0) and is_reverse=false: multiplies by KPlus_on_DLRtau[:,l]
- For KMinus(l) (index=-l, l>0) and is_reverse=false: multiplies by KMinus_on_DLRtau[:,l]
- When is_reverse=true, the KPlus/KMinus selection is swapped

The XOR logic `(index > 0) ⊻ is_reverse` determines which K to use:
- Positive index, not reverse → KPlus
- Negative index, not reverse → KMinus
- Positive index, reverse → KMinus
- Negative index, reverse → KPlus
"""
@inline function apply_hybridization_inplace!(temp_buffer::Array{T,3}, target::Array{T,3}, hyb::HybridizationState, model_data::ModelData{T}; is_reverse::Bool=false) where {T<:Number}
    index = hyb.index
    if index != 0  # Not NoHyb
        # XOR determines if we should use KPlus: (positive and not reverse) OR (negative and reverse)
        use_kplus = (index > 0) ⊻ is_reverse
        l = abs(index)
        vec = use_kplus ? @view(model_data.hyb.KPlus_on_DLRtau[:, l]) :
                          @view(model_data.hyb.KMinus_on_DLRtau[:, l])
        _multiplication_13!(temp_buffer, vec, target)
        copyto!(target, temp_buffer)
    end
    # For NoHyb, do nothing (target unchanged)
end

"""
    apply_hybridization(hyb, result, model_data; is_reverse=false)

Allocating version: creates a new array for the result.

# Arguments
- `hyb::HybridizationState`: Hybridization state
- `result::Array{Float64,3}`: Input array
- `model_data::ModelData`: Model data containing K± values
- `is_reverse::Bool=false`: When true, swaps KPlus ↔ KMinus

# Returns
- New array with hybridization applied, or original array if NoHyb
"""
@inline function apply_hybridization(hyb::HybridizationState, result::Array{Float64,3}, model_data::ModelData; is_reverse::Bool=false)
    index = hyb.index
    if index != 0  # Not NoHyb
        # XOR determines if we should use KPlus: (positive and not reverse) OR (negative and reverse)
        use_kplus = (index > 0) ⊻ is_reverse
        l = abs(index)
        return use_kplus ? _multiplication_13(model_data.hyb.KPlus_on_DLRtau[:, l], result) :
                           _multiplication_13(model_data.hyb.KMinus_on_DLRtau[:, l], result)
    else
        return result
    end
end
