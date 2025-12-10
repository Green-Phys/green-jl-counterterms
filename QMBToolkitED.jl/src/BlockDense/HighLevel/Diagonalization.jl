"""
Block-diagonal eigenvalue solver for block-dense operators.

For operators with flux=0 (like Hamiltonians), each sector can be diagonalized
independently, providing massive speedup over full diagonalization.
"""

"""
    BlockEigenResult

Result of block-diagonal eigenvalue decomposition.

Fields:
- `sector_eigenvalues::Vector{Vector{Float64}}`: Eigenvalues for each sector
- `sector_eigenvectors::Vector{Matrix{ComplexF64}}`: Eigenvectors for each sector
- `sector_indices::Vector{Int}`: Sector index for each eigenvalue (for sorting)
- `basis::BlockDenseBasis`: The basis used

The full eigenvalue spectrum is the concatenation of all sector eigenvalues.
"""
struct BlockEigenResult
    sector_eigenvalues::Vector{Vector{Float64}}
    sector_eigenvectors::Vector{Matrix{ComplexF64}}
    sector_indices::Vector{Int}
    basis::BlockDenseBasis
end

"""
    get_full_eigenvalues(result::BlockEigenResult) -> Vector{Float64}

Get all eigenvalues from all sectors concatenated.
"""
function get_full_eigenvalues(result::BlockEigenResult)
    return vcat(result.sector_eigenvalues...)
end

"""
    get_full_eigenvectors_sparse(result::BlockEigenResult) -> Matrix{ComplexF64}

Get all eigenvectors in the global (sparse) basis.

Each column is an eigenvector in the full Hilbert space.
Eigenvectors within a sector are placed at the appropriate global indices.
"""
function get_full_eigenvectors_sparse(result::BlockEigenResult)
    basis = result.basis
    total_dim = basis.total_dim
    num_evals = sum(length(evals) for evals in result.sector_eigenvalues)

    # Allocate full eigenvector matrix
    evecs_full = zeros(ComplexF64, total_dim, num_evals)

    # Compute sector starting indices in global space
    sector_starts = zeros(Int, length(basis.sectors))
    cumsum_dim = 0
    for (i, sector) in enumerate(basis.sectors)
        sector_starts[i] = cumsum_dim
        cumsum_dim += sector.dim
    end

    # Fill in eigenvectors sector by sector
    eval_col = 1
    for (sector_idx, sector_evecs) in enumerate(result.sector_eigenvectors)
        sector_start = sector_starts[sector_idx]
        sector_dim = basis.sectors[sector_idx].dim

        for col in 1:size(sector_evecs, 2)
            # Place this eigenvector in the global space
            evecs_full[(sector_start+1):(sector_start+sector_dim), eval_col] = sector_evecs[:, col]
            eval_col += 1
        end
    end

    return evecs_full
end

"""
    solve_hamiltonian_blockdense(H::BlockDenseOperator) -> BlockEigenResult

Diagonalize a block-dense Hamiltonian efficiently by processing each sector independently.

This function leverages the block-diagonal structure of the Hamiltonian (implied by flux=0)
to massive reduce computational cost compared to full diagonalization.

Arguments:
- `H::BlockDenseOperator`: Hamiltonian (must have flux=0)

Returns:
- `BlockEigenResult`: Structure containing eigenvalues and eigenvectors per sector.
  This result object is iterable, allowing destructuring into `(evals, evecs)` for 
  backward compatibility, where `evals` and `evecs` are the full (dense) spectrum.

Requirements:
- H must have flux = 0 (block-diagonal in the chosen basis)
- H should be Hermitian

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c = get_annihilators_blockdense(basis)
H = sum(c[i]' * c[i] for i in 1:2)

# Get result object (recommended for efficient post-processing)
result = solve_hamiltonian_blockdense(H)

# Calculate Green's function efficiently
G = calculate_green_tau_blockdense(result, c[1], c[1]', 10.0, 100)

# Or get full spectrum (less efficient)
evals, evecs = result
```

Performance:
For a system with n sectors of dimension d each, this is O(n * d^3) instead
of O((n*d)^3), a speedup of ~n^2 for equal-sized sectors.
"""
function solve_hamiltonian_blockdense(H::BlockDenseOperator)
    basis = H.basis

    # Get concrete quantum number type from first element
    QN_type = typeof(basis.site_qns[1])

    # Verify flux is zero (Hamiltonian is block-diagonal)
    if H.flux.delta_qn != zero_qn(QN_type)
        error("Hamiltonian must have zero flux for efficient block diagonalization. " *
              "Got flux = $(H.flux.delta_qn)")
    end

    num_sectors = length(basis.sectors)
    sector_eigenvalues = Vector{Vector{Float64}}(undef, num_sectors)
    sector_eigenvectors = Vector{Matrix{ComplexF64}}(undef, num_sectors)
    sector_indices = Int[]

    # Diagonalize each sector independently
    for (sector_idx, sector) in enumerate(basis.sectors)
        dim = sector.dim

        # Extract block for this sector (diagonal block)
        if haskey(H.blocks, (sector_idx, sector_idx))
            block = H.blocks[(sector_idx, sector_idx)]

            # Diagonalize this block
            # Make sure it's Hermitian
            sol = eigen(Hermitian(block))

            sector_eigenvalues[sector_idx] = sol.values
            sector_eigenvectors[sector_idx] = sol.vectors
        else
            # No block means this sector has zero energy
            # (e.g., vacuum sector for interaction-only Hamiltonian)
            sector_eigenvalues[sector_idx] = zeros(Float64, dim)
            sector_eigenvectors[sector_idx] = Matrix{ComplexF64}(I, dim, dim)
        end

        # Track which sector each eigenvalue belongs to
        append!(sector_indices, fill(sector_idx, dim))
    end

    return BlockEigenResult(
        sector_eigenvalues,
        sector_eigenvectors,
        sector_indices,
        basis
    )
end
