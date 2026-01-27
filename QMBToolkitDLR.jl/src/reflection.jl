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

TODO: implement the more efficient version working directly in DLR representation
"""
function do_reflection_tau2tau(
    dlr_grid::DLRGrid,
    H_tau::AbstractArray{T,3},
) where {T<:Number}
    H_dlr = tau2dlr(dlr_grid, H_tau, dlr_grid.τ; axis = 3)
    return do_reflection_dlr2tau(dlr_grid, H_dlr)
end
