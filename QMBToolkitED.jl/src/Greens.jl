"""
Green's function calculations for quantum many-body systems.

Exploits block-diagonal structure of the Hamiltonian and operators to compute
Green's functions efficiently without reconstructing the full dense matrices.
"""

"""
    _green_setup(result, c_i, c_j, beta) -> (basis, sector_diags, evec_blocks, min_E, Z)

Shared setup for Green's function calculations.
Verifies operator compatibility, extracts eigensystem data,
finds the ground state energy, and computes the partition function.
"""
function _green_setup(
    result::EigenResult{T},
    c_i::Operator,
    c_j::Operator,
    beta::Real,
) where {T}
    basis = result.eigenvalues.basis

    # Verify operators are compatible with basis
    if c_i.basis !== basis || c_j.basis !== basis
        error("Operators must use the same basis as the diagonalization result")
    end

    sector_diags = result.eigenvalues.sector_diags
    evec_blocks = result.eigenvectors.blocks

    # Find global minimum energy
    min_E = Inf
    for (_, evals) in sector_diags
        if !isempty(evals)
            min_E = min(min_E, minimum(evals))
        end
    end
    if min_E == Inf
        min_E = 0.0
    end

    # Partition function
    Z = 0.0
    for (_, sector_evals) in sector_diags
        if !isempty(sector_evals)
            Z += sum(exp.(-beta .* (sector_evals .- min_E)))
        end
    end

    return basis, sector_diags, evec_blocks, min_E, Z
end

"""
    _green_contributions(basis, sector_diags, evec_blocks, min_E, c_i, c_j)

Iterate over sector pairs connected by c_i and c_j, transform operator blocks
to the eigenbasis, and return a vector of (evals_m, evals_n, M) tuples where
M_ba = C̃_i(b,a) * conj(C̃_j(b,a)) is the matrix element product.
"""
function _green_contributions(basis, sector_diags, evec_blocks, min_E, c_i, c_j)
    contributions = Tuple{Vector{Float64},Vector{Float64},Matrix{ComplexF64}}[]

    for (idx_n, sector_n) in enumerate(basis.sectors)
        if !haskey(sector_diags, idx_n)
            continue
        end

        qn_m = sector_n.qn + c_i.flux
        if !haskey(basis.qn_to_sector, qn_m)
            continue
        end
        idx_m = basis.qn_to_sector[qn_m]

        if !haskey(sector_diags, idx_m)
            continue
        end

        if !haskey(c_i.blocks, (idx_m, idx_n)) || !haskey(c_j.blocks, (idx_m, idx_n))
            continue
        end

        block_i = c_i.blocks[(idx_m, idx_n)]
        block_j = c_j.blocks[(idx_m, idx_n)]

        evals_n = sector_diags[idx_n] .- min_E
        evecs_n = evec_blocks[(idx_n, idx_n)]

        evals_m = sector_diags[idx_m] .- min_E
        evecs_m = evec_blocks[(idx_m, idx_m)]

        C_i_tilde = evecs_m' * block_i * evecs_n
        C_j_tilde = evecs_m' * block_j * evecs_n

        M = C_i_tilde .* conj.(C_j_tilde)

        push!(contributions, (evals_m, evals_n, M))
    end

    return contributions
end

"""
    calculate_green_tau(result, c_i, c_j, beta, tau_grid; parallel_info=nothing)

Calculate imaginary-time Green's function G_ij(τ) using EigenResult on a custom tau grid.
G_ij(τ) = -⟨T_τ c_i(τ) c_j^†(0)⟩

Arguments:
- `result::EigenResult`: Diagonalization result from `solve_hamiltonian`
- `c_i::Operator`: Annihilation operator i
- `c_j::Operator`: Annihilation operator j (will be conjugated)
- `beta::Real`: Inverse temperature
- `tau_grid::AbstractVector{<:Real}`: Custom tau grid points

Keyword Arguments:
- `parallel_info`: If provided as `(rank, nprocs, reduce_func)`, work is distributed
  across processes. `reduce_func(array)` should perform an allreduce sum operation.
  If `nothing` (default), runs serially.

Returns:
- `gtau`: Vector of G(τ) values at the specified tau points
"""
function calculate_green_tau(
    result::EigenResult{T},
    c_i::Operator,
    c_j::Operator,
    beta::Real,
    tau_grid::AbstractVector{<:Real};
    parallel_info = nothing,
) where {T}
    basis, sector_diags, evec_blocks, min_E, Z = _green_setup(result, c_i, c_j, beta)
    contributions = _green_contributions(basis, sector_diags, evec_blocks, min_E, c_i, c_j)

    n_tau = length(tau_grid)

    # Build flat list of work items: (contrib_idx, b, a)
    work_items = Tuple{Int,Int,Int}[]
    for (c_idx, (evals_m, evals_n, _)) in enumerate(contributions)
        for b = 1:length(evals_m)
            for a = 1:length(evals_n)
                push!(work_items, (c_idx, b, a))
            end
        end
    end

    # Determine which items this rank handles
    if parallel_info === nothing
        my_start, my_step = 1, 1
        do_reduce = false
    else
        rank, nprocs, _ = parallel_info
        my_start, my_step = rank + 1, nprocs
        do_reduce = true
    end

    # Compute local contribution
    gtau_local = zeros(ComplexF64, n_tau)
    for idx = my_start:my_step:length(work_items)
        c_idx, b, a = work_items[idx]
        evals_m, evals_n, M = contributions[c_idx]
        E_b = evals_m[b]
        E_a = evals_n[a]
        gtau_local .+= M[b, a] .* exp.(.-E_b .* (beta .- tau_grid) .- E_a .* tau_grid)
    end

    # Reduce across processes if parallel
    if do_reduce
        _, _, reduce_func = parallel_info
        gtau = reduce_func(gtau_local)
    else
        gtau = gtau_local
    end

    gtau ./= -Z
    return gtau
end

"""
    calculate_green_tau(result, c_i, c_j, beta, n_tau::Int)

Calculate imaginary-time Green's function G_ij(τ) using EigenResult on a uniform tau grid.
G_ij(τ) = -⟨T_τ c_i(τ) c_j^†(0)⟩

Arguments:
- `result::EigenResult`: Diagonalization result from `solve_hamiltonian`
- `c_i::Operator`: Annihilation operator i
- `c_j::Operator`: Annihilation operator j (will be conjugated)
- `beta::Real`: Inverse temperature
- `n_tau::Int`: Number of time points (uniform grid from 0 to beta)

Returns:
- `tau_grid`: Vector of time points
- `gtau`: Vector of G(τ) values
"""
function calculate_green_tau(
    result::EigenResult{T},
    c_i::Operator,
    c_j::Operator,
    beta::Real,
    n_tau::Int,
) where {T}
    tau_grid = range(0, beta, length = n_tau)
    gtau = calculate_green_tau(result, c_i, c_j, beta, tau_grid)
    return tau_grid, gtau
end

"""
    calculate_green_matsubara(result, c_i, c_j, beta, n_matsubara; parallel_info=nothing)

Calculate Matsubara Green's function G_ij(iω_n) using EigenResult.

Arguments:
- `result::EigenResult`: Diagonalization result
- `c_i::Operator`: Annihilation operator i
- `c_j::Operator`: Annihilation operator j
- `beta::Real`: Inverse temperature
- `n_matsubara::Int`: Number of frequencies

Keyword Arguments:
- `parallel_info`: If provided as `(rank, nprocs, reduce_func)`, work is distributed
  across processes. `reduce_func(array)` should perform an allreduce sum operation.
  If `nothing` (default), runs serially.

Returns:
- `iomega_n`: Vector of Matsubara frequencies
- `g_iwn`: Vector of G(iω_n) values
"""
function calculate_green_matsubara(
    result::EigenResult{T},
    c_i::Operator,
    c_j::Operator,
    beta::Real,
    n_matsubara::Int;
    parallel_info = nothing,
) where {T}
    basis, sector_diags, evec_blocks, min_E, Z = _green_setup(result, c_i, c_j, beta)
    contributions = _green_contributions(basis, sector_diags, evec_blocks, min_E, c_i, c_j)

    omega_n = (2 .* (0:(n_matsubara-1)) .+ 1) .* π ./ beta
    iomega_n = 1im .* omega_n

    # Build flat list of work items: (contrib_idx, b, a)
    work_items = Tuple{Int,Int,Int}[]
    for (c_idx, (evals_m, evals_n, _)) in enumerate(contributions)
        for b = 1:length(evals_m)
            for a = 1:length(evals_n)
                push!(work_items, (c_idx, b, a))
            end
        end
    end

    # Determine which items this rank handles
    if parallel_info === nothing
        my_start, my_step = 1, 1
        do_reduce = false
    else
        rank, nprocs, _ = parallel_info
        my_start, my_step = rank + 1, nprocs
        do_reduce = true
    end

    # Compute local contribution
    g_iwn_local = zeros(ComplexF64, n_matsubara)
    for idx = my_start:my_step:length(work_items)
        c_idx, b, a = work_items[idx]
        evals_m, evals_n, M = contributions[c_idx]
        E_m = evals_m[b]
        energy_diff = E_m - evals_n[a]
        boltzmann_factor = exp(-beta * E_m) + exp(-beta * evals_n[a])
        g_iwn_local .+= boltzmann_factor * M[b, a] ./ (iomega_n .+ energy_diff)
    end

    # Reduce across processes if parallel
    if do_reduce
        _, _, reduce_func = parallel_info
        g_iwn = reduce_func(g_iwn_local)
    else
        g_iwn = g_iwn_local
    end

    g_iwn ./= Z
    return iomega_n, g_iwn
end
