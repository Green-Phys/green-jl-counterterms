"""
Green's function calculations for dense matrices.
"""

"""
    calculate_green_tau_dense(evals, evecs, annh_ops, beta, n_tau, i, j)

Compute an element of the imaginary-time Green's function matrix G(τ) using dense matrices.
Specifically, it calculates -<c_i(τ) c_j^†(0)>.
"""
function calculate_green_tau_dense(
    evals::Vector{Float64},
    evecs::Matrix{ComplexF64},
    annh_ops::Vector{Matrix{ComplexF64}},
    beta::Real,
    n_tau::Int,
    i::Int,
    j::Int
)
    # Get annihilation operators for indices i and j
    c_i = annh_ops[i]
    c_j = annh_ops[j]
    c_j_dag = c_j'

    # Transform operators to the eigenbasis of the Hamiltonian
    c_i_tilde = evecs' * c_i * evecs
    c_j_dag_tilde = evecs' * c_j_dag * evecs

    # Shift energies so that ground state is at 0
    # Make a copy to avoid modifying the input
    local_evals = copy(evals)
    local_evals .-= minimum(local_evals)

    # Partition function
    # We need to compute Z here since we don't pass it in
    # Use the same shift_energy logic
    Z = sum(exp.(-beta .* local_evals))

    tau_grid = range(0, beta, length=n_tau)
    gtau = zeros(ComplexF64, n_tau)

    evolve(tau, e) = Diagonal(exp.(-tau .* e))

    for (k, t) in enumerate(tau_grid)
        D1 = evolve(beta - t, local_evals)
        D2 = evolve(t, local_evals)
        # Calculates Tr(exp(-(β-τ)H) c_i exp(-τH) c_j^†)
        gtau[k] = tr(D1 * c_i_tilde * D2 * c_j_dag_tilde)
    end

    gtau ./= -Z

    return tau_grid, gtau
end

"""
    calculate_green_matsubara_dense(evals, evecs, annh_ops, beta, n_matsubara, i, j)

Compute an element of the Matsubara Green's function G(iω_n) using dense matrices.
"""
function calculate_green_matsubara_dense(
    evals::Vector{Float64},
    evecs::Matrix{ComplexF64},
    annh_ops::Vector{Matrix{ComplexF64}},
    beta::Real,
    n_matsubara::Int,
    i::Int,
    j::Int
)
    nfock = size(evecs, 1)

    # Get annihilation operators for indices i and j
    c_i = annh_ops[i]
    c_j = annh_ops[j]
    c_j_dag = c_j'

    # Transform operators to the eigenbasis of the Hamiltonian
    c_i_tilde = evecs' * c_i * evecs
    c_j_dag_tilde = evecs' * c_j_dag * evecs

    # Shift energies so that ground state is at 0
    local_evals = copy(evals)
    local_evals .-= minimum(local_evals)

    # Partition function
    Z = sum(exp.(-beta .* local_evals))

    # Matsubara frequencies
    omega_n = (2 .* (0:(n_matsubara-1)) .+ 1) .* π ./ beta
    iomega_n = 1im .* omega_n

    g_iwn = zeros(ComplexF64, n_matsubara)

    for m in 1:nfock, n in 1:nfock
        energy_diff = local_evals[m] - local_evals[n]
        
        # <m|c_i|n> <n|c_j^†|m>
        matrix_element = c_i_tilde[m, n] * c_j_dag_tilde[n, m]
        
        if abs(matrix_element) > 1e-9
            boltzmann_factor = exp(-beta * local_evals[m]) + exp(-beta * local_evals[n])
            
            g_iwn .+= boltzmann_factor * matrix_element ./ (iomega_n .+ energy_diff)
        end
    end

    g_iwn ./= Z

    return iomega_n, g_iwn
end
