using Lehmann

function kernel_tau(tau, omega, beta)
    @assert tau >= -beta && tau <= beta "tau must be in [-beta, beta]"
    if tau >= 0
        if omega > 0
            return exp(-tau * omega)/(1 + exp(-beta * omega))
        else
            return exp(-(tau - beta) * omega)/(1 + exp(beta * omega))   
        end
    else
        if omega > 0
            return -1 * exp(-(tau + beta) * omega)/(1 + exp(-beta * omega))
        else
            return -1 * exp(-tau * omega)/(1 + exp(beta * omega))   
        end
    end
end

"""
    Calculate \$\\int_{0}^{\\tau} d\\tau' F(\\tau-\\tau') G(\\tau')\$
    where F and G are represented in the DLR basis, which are Arrays of size (r,o1,o2).
    return the results on the DLR tau grid as an Array of size (r,o1,o2).
"""
# TODO: optimize the implementation (cache kernel evaluations)
function do_convolution_dlr2tau(dlr_grid::DLRGrid, F_dlr::Array{Float64,3}, G_dlr::Array{Float64,3})
    r = size(F_dlr, 1)
    o1 = size(F_dlr, 2)
    o2 = size(F_dlr, 3)

    H1_dlr = zeros(Float64, r, o1, o2)
    for j in 1:r
        H1_dlr[j, :, :] = kernel_tau(0.0, dlr_grid.omega[j], dlr_grid.beta) .* (F_dlr[j, :, :] * G_dlr[j, :, :])
    end

    H2_dlr = zeros(Float64, r, o1, o2)
    for j in 1:r
        term1 = zeros(Float64, o1, o2)
        term2 = zeros(Float64, o1, o2)
        for k in 1:r
            if k != j
                coeff = kernel_tau(0, dlr_grid.omega[k], dlr_grid.beta) / (dlr_grid.omega[k] - dlr_grid.omega[j])
                term1 .+= coeff .* F_dlr[k, :, :]
                term2 .+= coeff .* G_dlr[k, :, :]
            end
        end
        term1 *= G_dlr[j, :, :]
        term2 = F_dlr[j, :, :] * term2
        H2_dlr[j, :, :] = term1 .+ term2
    end

    H1_tau = dlr2tau(dlr_grid, H1_dlr, dlr_grid.tau)
    H2_tau = dlr2tau(dlr_grid, H2_dlr, dlr_grid.tau)
    H_tau = dlr_grid.tau .* H1_tau .+ H2_tau
    return H_tau
end

""" 
    Perform reflection operation: F(τ) = H(β - τ)
    where F is represented in the DLR basis, which is an Array of size (r,o1,o2).
    return the results on the DLR tau grid as an Array of size (r, o1, o2).
"""
function do_reflection_dlr2tau(dlr_grid::DLRGrid, H_dlr::Array{Float64,3})
    tau_grid_reflected = dlr_grid.beta .- dlr_grid.tau
    F_tau = dlr2tau(dlr_grid, H_dlr, tau_grid_reflected)
    return F_tau
end


# TODO: implement the more efficient version working directly in DLR representation
function do_reflection_tau2tau(dlr_grid::DLRGrid, H_tau::Array{Float64,3})
    H_dlr = tau2dlr(dlr_grid, H_tau, dlr_grid.tau)
    return do_reflection_dlr2tau(dlr_grid, H_dlr)
end


"""
    Perform multiplication in DLR representation: H(tau) = F(tau) * G(tau)
"""
function do_multiplication_tau2tau(F_tau::Array{Float64,3}, G_tau::Array{Float64,3})
    H_tau = zeros(Float64, size(F_tau))
    for k in 1:size(F_tau, 1)
        H_tau[k, :, :] = F_tau[k, :, :] * G_tau[k, :, :]
    end
    return H_tau
end