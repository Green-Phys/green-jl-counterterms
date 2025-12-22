"""
Green's function calculations for dense matrices.
"""

"""
    calculate_green_tau_dense(result, c_i, c_j, beta, n_tau)

Compute an element of the imaginary-time Green's function matrix G(τ).
Specifically, it calculates -<c_i(τ) c_j^†(0)>.

Arguments:
- `result::DenseEigenResult`: Eigenvalue decomposition result
- `c_i::DenseOperator`: Annihilation operator for orbital i
- `c_j::DenseOperator`: Annihilation operator for orbital j
- `beta::Real`: Inverse temperature
- `n_tau::Int`: Number of imaginary time points

Returns:
- `tau_grid`: Array of τ values from 0 to β
- `gtau`: Green's function values at each τ
"""
function calculate_green_tau_dense(
    result::DenseEigenResult,
    c_i::DenseOperator,
    c_j::DenseOperator,
    beta::Real,
    n_tau::Int
)
    evals = result.eigenvalues.diag
    evecs = result.eigenvectors.data

    # Get operator matrices
    c_i_mat = c_i.data
    c_j_dag_mat = c_j.data'

    # Transform operators to the eigenbasis of the Hamiltonian
    c_i_tilde = evecs' * c_i_mat * evecs
    c_j_dag_tilde = evecs' * c_j_dag_mat * evecs

    # Shift energies so that ground state is at 0
    local_evals = copy(evals)
    local_evals .-= minimum(local_evals)

    # Partition function
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
    calculate_green_matsubara_dense(result, c_i, c_j, beta, n_matsubara)

Compute an element of the Matsubara Green's function G(iω_n).

Arguments:
- `result::DenseEigenResult`: Eigenvalue decomposition result
- `c_i::DenseOperator`: Annihilation operator for orbital i
- `c_j::DenseOperator`: Annihilation operator for orbital j
- `beta::Real`: Inverse temperature
- `n_matsubara::Int`: Number of Matsubara frequencies

Returns:
- `iomega_n`: Array of Matsubara frequencies iω_n
- `g_iwn`: Green's function values at each frequency
"""
function calculate_green_matsubara_dense(
    result::DenseEigenResult,
    c_i::DenseOperator,
    c_j::DenseOperator,
    beta::Real,
    n_matsubara::Int
)
    evals = result.eigenvalues.diag
    evecs = result.eigenvectors.data
    nfock = size(evecs, 1)

    # Get operator matrices
    c_i_mat = c_i.data
    c_j_dag_mat = c_j.data'

    # Transform operators to the eigenbasis of the Hamiltonian
    c_i_tilde = evecs' * c_i_mat * evecs
    c_j_dag_tilde = evecs' * c_j_dag_mat * evecs

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
