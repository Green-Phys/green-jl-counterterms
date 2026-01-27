"""
Reflection and multiplication operations for dense representation.

Layout: (dim1, dim2, n_tau) for tau-space, (dim1, dim2, r) for DLR representation.
"""

using Lehmann

"""
    do_reflection_dlr2tau(dlr_grid::DLRGrid, H_dlr::AbstractArray{T,3}) where T<:Number

Perform reflection operation: G(τ) = H(β - τ).

# Arguments
- `dlr_grid`: DLR grid used for the transformation.
- `H_dlr`: Input function H in DLR representation, size (o1, o2, r).

# Returns
- `G_tau`: Result G evaluated on the DLR tau grid, size (o1, o2, n_tau).
"""
function do_reflection_dlr2tau(
    dlr_grid::DLRGrid,
    H_dlr::AbstractArray{T,3},
) where {T<:Number}
    tau_grid_reflected = dlr_grid.β .- dlr_grid.τ
    F_tau = dlr2tau(dlr_grid, H_dlr, tau_grid_reflected; axis = 3)
    return F_tau
end

"""
    do_reflection_tau2tau(dlr_grid::DLRGrid, H_tau::AbstractArray{T,3}) where T<:Number

Perform reflection operation on tau-space data.
First transforms to DLR, then applies reflection.

NOTE: This allocating version exists for convenience. For hot loops, use do_reflection_tau2tau!
"""
function do_reflection_tau2tau(
    dlr_grid::DLRGrid,
    H_tau::AbstractArray{T,3},
) where {T<:Number}
    H_dlr = tau2dlr(dlr_grid, H_tau, dlr_grid.τ; axis = 3)
    return do_reflection_dlr2tau(dlr_grid, H_dlr)
end

"""
    do_reflection_tau2tau!(dest::AbstractArray{T,3}, dlr_buffer::AbstractArray{T,3},
                           dlr_grid::DLRGrid, src::AbstractArray{T,3},
                           workspace::DLRWorkspace{T};
                           τ_reflected::Union{Nothing, AbstractVector{Float64}}=nothing) where T<:Number

In-place reflection operation: dest = src(β - τ).
Uses pre-allocated dlr_buffer and DLRWorkspace to avoid allocations.

# Arguments
- `dest`: Output array in tau space (dim1, dim2, n_tau)
- `dlr_buffer`: Pre-allocated buffer for DLR coefficients (dim1, dim2, n_dlr)
- `dlr_grid`: DLR grid
- `src`: Input array in tau space (dim1, dim2, n_tau)
- `workspace`: DLRWorkspace for in-place operations
- `τ_reflected`: Optional pre-computed β - τ grid (avoids allocation if provided)

# Notes
For hot loops, pass a pre-computed `τ_reflected` to avoid allocations.
This can be obtained from `ConvolutionWorkspace.τ_reflected`.
"""
function do_reflection_tau2tau!(
    dest::AbstractArray{T,3},
    dlr_buffer::AbstractArray{T,3},
    dlr_grid::DLRGrid,
    src::AbstractArray{T,3},
    workspace::DLRWorkspace{T};
    τ_reflected::Union{Nothing,AbstractVector{Float64}} = nothing,
) where {T<:Number}
    # Transform to DLR coefficients (in-place)
    tau2dlr!(dlr_buffer, dlr_grid, src, workspace)

    # Use pre-computed reflected grid if provided, otherwise compute it
    τ_refl = isnothing(τ_reflected) ? dlr_grid.β .- dlr_grid.τ : τ_reflected

    # Use dlr2tau with reflected grid
    dlr2tau!(dest, dlr_grid, dlr_buffer, τ_refl, workspace)

    return dest
end
