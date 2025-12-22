"""
Green's function calculations for block-dense operators.

Exploits block-diagonal structure of the Hamiltonian and operators to compute
Green's functions efficiently without reconstructing the full dense matrices.
"""

"""
    calculate_green_tau_blockdense(result, c_i, c_j, beta, n_tau)

Calculate imaginary-time Green's function G_ij(τ) using BlockEigenResult.
G_ij(τ) = -⟨T_τ c_i(τ) c_j^†(0)⟩

Arguments:
- `result::BlockEigenResult`: Diagonalization result from `solve_hamiltonian_blockdense`
- `c_i::BlockDenseOperator`: Annihilation operator i
- `c_j::BlockDenseOperator`: Annihilation operator j (will be conjugated)
- `beta::Real`: Inverse temperature
- `n_tau::Int`: Number of time points

Returns:
- `tau_grid`: Vector of time points
- `gtau`: Vector of G(τ) values
"""
function calculate_green_tau_blockdense(
    result::BlockEigenResult,
    c_i::BlockDenseOperator,
    c_j::BlockDenseOperator,
    beta::Real,
    n_tau::Int
)
    basis = result.eigenvalues.basis

    # Verify operators are compatible with basis
    if c_i.basis !== basis || c_j.basis !== basis
        error("Operators must use the same basis as the diagonalization result")
    end

    # Calculate Z and E_min from the diagonal eigenvalues
    sector_diags = result.eigenvalues.sector_diags
    evec_blocks = result.eigenvectors.blocks

    # Find global minimum energy safely
    min_E = Inf
    for (_, evals) in sector_diags
        if !isempty(evals)
            min_E = min(min_E, minimum(evals))
        end
    end
    if min_E == Inf
        min_E = 0.0 # Empty system?
    end

    Z = 0.0
    for (_, sector_evals) in sector_diags
        if !isempty(sector_evals)
            Z += sum(exp.(-beta .* (sector_evals .- min_E)))
        end
    end

    tau_grid = range(0, beta, length=n_tau)
    gtau = zeros(ComplexF64, n_tau)

    # Iterate over all source sectors S_n
    for (idx_n, sector_n) in enumerate(basis.sectors)
        # Skip if no eigenvalues for this sector
        if !haskey(sector_diags, idx_n)
            continue
        end

        # Determine target sector S_m for c_i |n> -> |m>
        # Q_m = Q_n + flux(c_i)
        qn_m = combine_qn(sector_n.qn, c_i.flux.delta_qn)
        if !haskey(basis.qn_to_sector, qn_m)
            continue
        end
        idx_m = basis.qn_to_sector[qn_m]

        # Skip if no eigenvalues for target sector
        if !haskey(sector_diags, idx_m)
            continue
        end

        # Check if blocks exist
        if !haskey(c_i.blocks, (idx_m, idx_n)) || !haskey(c_j.blocks, (idx_m, idx_n))
            continue
        end

        block_i = c_i.blocks[(idx_m, idx_n)] # Dim: dim_m x dim_n
        block_j = c_j.blocks[(idx_m, idx_n)] # Dim: dim_m x dim_n

        # Get eigensystems (shifted energies)
        evals_n = sector_diags[idx_n] .- min_E
        evecs_n = evec_blocks[(idx_n, idx_n)] # dim_n x dim_n

        evals_m = sector_diags[idx_m] .- min_E
        evecs_m = evec_blocks[(idx_m, idx_m)] # dim_m x dim_m

        # Transform blocks to eigenbasis
        # \tilde{C}_i = U_m' * C_i * U_n
        # \tilde{C}_j = U_m' * C_j * U_n
        # Note: We use C_j, not C_j^dag, because the formula uses <n|c_j^dag|m> = conj(<m|c_j|n>)

        C_i_tilde = evecs_m' * block_i * evecs_n
        C_j_tilde = evecs_m' * block_j * evecs_n

        # Matrix element product M_ba = (C_i_tilde)_ba * conj((C_j_tilde)_ba)
        M = C_i_tilde .* conj.(C_j_tilde)

        # Sum over states a (in S_n) and b (in S_m)
        # term_{ba} = exp(-beta * E_b) * exp(tau * (E_b - E_a)) * M_{ba}

        dim_m = length(evals_m)
        dim_n = length(evals_n)

        for b in 1:dim_m
            E_b = evals_m[b]
            factor_beta = exp(-beta * E_b)

            for a in 1:dim_n
                E_a = evals_n[a]
                val = M[b, a]

                if abs(val) > 1e-12 # Optimization: skip small elements
                    diff_E = E_b - E_a

                    for k in 1:n_tau
                        t = tau_grid[k]
                        gtau[k] += factor_beta * exp(t * diff_E) * val
                    end
                end
            end
        end
    end

    gtau ./= -Z
    return tau_grid, gtau
end

"""
    calculate_green_matsubara_blockdense(result, c_i, c_j, beta, n_matsubara)

Calculate Matsubara Green's function G_ij(iω_n) using BlockEigenResult.

Arguments:
- `result::BlockEigenResult`: Diagonalization result
- `c_i::BlockDenseOperator`: Annihilation operator i
- `c_j::BlockDenseOperator`: Annihilation operator j
- `beta::Real`: Inverse temperature
- `n_matsubara::Int`: Number of frequencies

Returns:
- `iomega_n`: Vector of Matsubara frequencies
- `g_iwn`: Vector of G(iω_n) values
"""
function calculate_green_matsubara_blockdense(
    result::BlockEigenResult,
    c_i::BlockDenseOperator,
    c_j::BlockDenseOperator,
    beta::Real,
    n_matsubara::Int
)
    basis = result.eigenvalues.basis

    # Verify operators are compatible with basis
    if c_i.basis !== basis || c_j.basis !== basis
        error("Operators must use the same basis as the diagonalization result")
    end

    # Calculate Z and E_min from the diagonal eigenvalues
    sector_diags = result.eigenvalues.sector_diags
    evec_blocks = result.eigenvectors.blocks

    # Find global minimum energy safely
    min_E = Inf
    for (_, evals) in sector_diags
        if !isempty(evals)
            min_E = min(min_E, minimum(evals))
        end
    end
    if min_E == Inf
        min_E = 0.0
    end

    Z = 0.0
    for (_, sector_evals) in sector_diags
        if !isempty(sector_evals)
            Z += sum(exp.(-beta .* (sector_evals .- min_E)))
        end
    end

    omega_n = (2 .* (0:(n_matsubara-1)) .+ 1) .* π ./ beta
    iomega_n = 1im .* omega_n
    g_iwn = zeros(ComplexF64, n_matsubara)

    # Iterate over all source sectors S_n
    for (idx_n, sector_n) in enumerate(basis.sectors)
        # Skip if no eigenvalues for this sector
        if !haskey(sector_diags, idx_n)
            continue
        end

        # Target sector S_m for c_i
        qn_m = combine_qn(sector_n.qn, c_i.flux.delta_qn)
        if !haskey(basis.qn_to_sector, qn_m)
            continue
        end
        idx_m = basis.qn_to_sector[qn_m]

        # Skip if no eigenvalues for target sector
        if !haskey(sector_diags, idx_m)
            continue
        end

        # Check blocks
        if !haskey(c_i.blocks, (idx_m, idx_n)) || !haskey(c_j.blocks, (idx_m, idx_n))
            continue
        end

        block_i = c_i.blocks[(idx_m, idx_n)]
        block_j = c_j.blocks[(idx_m, idx_n)]

        # Get eigensystems
        evals_n = sector_diags[idx_n] .- min_E
        evecs_n = evec_blocks[(idx_n, idx_n)]

        evals_m = sector_diags[idx_m] .- min_E
        evecs_m = evec_blocks[(idx_m, idx_m)]

        # Transform blocks
        C_i_tilde = evecs_m' * block_i * evecs_n
        C_j_tilde = evecs_m' * block_j * evecs_n

        M = C_i_tilde .* conj.(C_j_tilde)

        dim_m = length(evals_m)
        dim_n = length(evals_n)

        for b in 1:dim_m # state m
            E_m = evals_m[b]
            for a in 1:dim_n # state n
                E_n = evals_n[a]

                matrix_element = M[b, a]

                if abs(matrix_element) > 1e-12
                    energy_diff = E_m - E_n
                    boltzmann_factor = exp(-beta * E_m) + exp(-beta * E_n)

                    # Vectorized update for all frequencies
                    g_iwn .+= boltzmann_factor * matrix_element ./ (iomega_n .+ energy_diff)
                end
            end
        end
    end

    g_iwn ./= Z
    return iomega_n, g_iwn
end
