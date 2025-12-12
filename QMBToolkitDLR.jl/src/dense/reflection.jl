"""
Reflection and multiplication operations for dense representation.
"""

using Lehmann

"""
    do_reflection_dlr2tau(dlr_grid::DLRGrid, H_dlr::Array{T,3}) where T<:Number

Perform reflection operation: G(τ) = H(β - τ).

# Arguments
- `dlr_grid`: DLR grid used for the transformation.
- `H_dlr`: Input function H in DLR representation, size (r, o1, o2).

# Returns
- `G_tau`: Result G evaluated on the DLR tau grid, size (n_tau, o1, o2).
"""
function do_reflection_dlr2tau(dlr_grid::DLRGrid, H_dlr::Array{T,3}) where T<:Number
    tau_grid_reflected = dlr_grid.beta .- dlr_grid.tau
    F_tau = dlr2tau(dlr_grid, H_dlr, tau_grid_reflected)
    return F_tau
end

"""
    do_reflection_tau2tau(dlr_grid::DLRGrid, H_tau::Array{T,3}) where T<:Number

Perform reflection operation on tau-space data.
First transforms to DLR, then applies reflection.

TODO: implement the more efficient version working directly in DLR representation
"""
function do_reflection_tau2tau(dlr_grid::DLRGrid, H_tau::Array{T,3}) where T<:Number
    H_dlr = tau2dlr(dlr_grid, H_tau, dlr_grid.tau)
    return do_reflection_dlr2tau(dlr_grid, H_dlr)
end

"""
    do_multiplication_tau2tau(F_tau::Array{T,3}, G_tau::Array{T,3}) where T<:Number

Perform element-wise matrix multiplication in tau-space: H(tau) = F(tau) * G(tau)
"""
function do_multiplication_tau2tau(F_tau::Array{T,3}, G_tau::Array{T,3}) where T<:Number
    H_tau = zeros(T, size(F_tau))
    for k in 1:size(F_tau, 1)
        H_tau[k, :, :] = F_tau[k, :, :] * G_tau[k, :, :]
    end
    return H_tau
end