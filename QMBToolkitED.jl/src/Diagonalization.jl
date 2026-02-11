"""
Eigenvalue solver for block-dense operators.

For operators with flux=0 (like Hamiltonians), each sector can be diagonalized
independently, providing massive speedup over full diagonalization.
"""

"""
    EigenResult{T}

Result of block-diagonal eigenvalue decomposition.

Type Parameters:
- `T<:Number`: Element type of eigenvectors (Float64 for real, ComplexF64 for complex)

Fields:
- `eigenvalues::DiagonalOperator`: Eigenvalues as a diagonal operator (always Float64)
- `eigenvectors::Operator{T}`: Eigenvectors as a block-diagonal unitary operator

The eigenvectors operator has flux=0 and stores the unitary transformation matrix
for each sector. Each block (sector_idx, sector_idx) contains the eigenvectors
for that sector as columns.

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
c = annihilators(basis)
H = sum(c[i]' * c[i] for i in 1:2)

result = solve_hamiltonian(H)

# Access eigenvalues for sector 2
E_sector2 = result.eigenvalues.sector_diags[2]

# Access eigenvectors for sector 2
U_sector2 = result.eigenvectors.blocks[(2, 2)]
```
"""
struct EigenResult{T<:Number}
    eigenvalues::DiagonalOperator
    eigenvectors::Operator{T}
end

"""
    solve_hamiltonian(H::Operator; parallel_info=nothing) -> EigenResult

Diagonalize a Hamiltonian efficiently by processing each sector independently.

This function leverages the block-diagonal structure of the Hamiltonian (implied by flux=0)
to massively reduce computational cost compared to full diagonalization.

Arguments:
- `H::Operator`: Hamiltonian (must have flux=0)

Keyword Arguments:
- `parallel_info`: If provided as `(rank, nprocs, gather_func)`, sectors are distributed
  across processes. `gather_func(local_sector_diags, local_evec_blocks)` should gather
  and merge results from all processes, returning `(merged_sector_diags, merged_evec_blocks)`.
  If `nothing` (default), runs serially.

Returns:
- `EigenResult`: Structure containing eigenvalues and eigenvectors as operator types.

Requirements:
- H must have flux = 0 (block-diagonal in the chosen basis)
- H should be Hermitian

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
c = annihilators(basis)
H = sum(c[i]' * c[i] for i in 1:2)

# Get result object
result = solve_hamiltonian(H)

# Access eigenvalues per sector
for (idx, evals) in result.eigenvalues.sector_diags
    println("Sector ", idx, ": ", evals)
end
```

Performance:
For a system with n sectors of dimension d each, this is O(n * d^3) instead
of O((n*d)^3), a speedup of ~n^2 for equal-sized sectors.
"""
function solve_hamiltonian(H::Operator{T}; parallel_info = nothing) where {T}
    basis = H.basis

    # Get concrete quantum number type from first element
    QN_type = typeof(basis.flavor_qns[1])

    # Verify flux is zero (Hamiltonian is block-diagonal)
    if H.flux != zero(QN_type)
        error(
            "Hamiltonian must have zero flux for efficient block diagonalization. " *
            "Got flux = $(H.flux)",
        )
    end

    # Determine which sectors this rank handles
    n_sectors = length(basis.sectors)
    if parallel_info === nothing
        my_start, my_step = 1, 1
        do_gather = false
    else
        rank, nprocs, _ = parallel_info
        my_start, my_step = rank + 1, nprocs
        do_gather = true
    end

    # Storage for eigenvalues and eigenvectors (initially complex)
    local_sector_diags = Dict{Int,Vector{Float64}}()
    local_evec_blocks = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()

    # Diagonalize sectors assigned to this rank
    for sector_idx = my_start:my_step:n_sectors
        sector = basis.sectors[sector_idx]
        dim = sector.dim

        # Extract block for this sector (diagonal block)
        if haskey(H.blocks, (sector_idx, sector_idx))
            block = H.blocks[(sector_idx, sector_idx)]

            # Diagonalize this block
            # Make sure it's Hermitian
            sol = eigen(Hermitian(block))

            local_sector_diags[sector_idx] = sol.values
            local_evec_blocks[(sector_idx, sector_idx)] = sol.vectors
        else
            # No block means this sector has zero energy
            # (e.g., vacuum sector for interaction-only Hamiltonian)
            local_sector_diags[sector_idx] = zeros(Float64, dim)
            local_evec_blocks[(sector_idx, sector_idx)] = Matrix{ComplexF64}(I, dim, dim)
        end
    end

    # Gather results from all processes if parallel
    if do_gather
        _, _, gather_func = parallel_info
        sector_diags, evec_blocks = gather_func(local_sector_diags, local_evec_blocks)
    else
        sector_diags = local_sector_diags
        evec_blocks = local_evec_blocks
    end

    # Create eigenvalues operator
    zf = zero(QN_type)
    eigenvalues = DiagonalOperator(basis, zf, sector_diags)

    # Check if all eigenvectors are real (imaginary parts are zero)
    all_real = all(all(imag.(block) .== 0) for block in values(evec_blocks))

    if all_real
        # Convert to Float64 eigenvectors
        real_evec_blocks =
            Dict{Tuple{Int,Int},Matrix{Float64}}(k => real.(v) for (k, v) in evec_blocks)
        eigenvectors = Operator{Float64}(basis, zf, real_evec_blocks)
        return EigenResult{Float64}(eigenvalues, eigenvectors)
    else
        eigenvectors = Operator{ComplexF64}(basis, zf, evec_blocks)
        return EigenResult{ComplexF64}(eigenvalues, eigenvectors)
    end
end
