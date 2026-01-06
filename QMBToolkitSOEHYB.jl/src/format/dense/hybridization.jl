"""
Hybridization application operations for dense representation.

Functions for applying SOE-decomposed hybridization functions K±(τ) to propagators.
Works with DensePropagator wrapper types.
"""

"""
    apply_hybridization_inplace_dense!(temp_buffer, target, hyb, model_data; is_reverse=false)

Apply hybridization function to target propagator in-place, updating target with the result.
Uses pre-allocated temp_buffer to avoid allocations.

# Arguments
- `temp_buffer::DensePropagator{T}`: Pre-allocated buffer for intermediate results
- `target::DensePropagator{T}`: Propagator to be modified (τ, orbital1, orbital2)
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
@inline function apply_hybridization_inplace_dense!(
        temp_buffer::DensePropagator{T}, target::DensePropagator{T}, hyb::HybridizationState,
        model_data::ModelData{T}; is_reverse::Bool = false,) where {T <: Number}
    index = hyb.index
    if index != 0  # Not NoHyb
        # XOR determines if we should use KPlus: (positive and not reverse) OR (negative and reverse)
        use_kplus = (index > 0) ⊻ is_reverse
        l = abs(index)
        vec = use_kplus ? @view(model_data.hyb.KPlus_on_DLRtau[:, l]) :
              @view(model_data.hyb.KMinus_on_DLRtau[:, l])
        _multiplication_13_dense!(temp_buffer, vec, target)
        copyto!(target, temp_buffer)
    end
    # For NoHyb, do nothing (target unchanged)
end

#=============================================================================
# Interface Implementations
=============================================================================#

# Hybridization application - interface version
function apply_hybridization_inplace!(
        temp_buffer::DensePropagator{T}, target::DensePropagator{T}, hyb::HybridizationState,
        model_data::ModelData{T}; is_reverse::Bool = false,) where {T}
    apply_hybridization_inplace_dense!(
        temp_buffer, target, hyb, model_data; is_reverse = is_reverse,)
end
