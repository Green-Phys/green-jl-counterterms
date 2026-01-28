"""
Eigenvalue solver for block-dense operators.

For operators with flux=0 (like Hamiltonians), each sector can be diagonalized
independently, providing massive speedup over full diagonalization.
"""

"""
    EigenResult

Result of block-diagonal eigenvalue decomposition.

Fields:
- `eigenvalues::DiagonalOperator`: Eigenvalues as a diagonal operator
- `eigenvectors::Operator`: Eigenvectors as a block-diagonal unitary operator

The eigenvectors operator has flux=0 and stores the unitary transformation matrix
for each sector. Each block (sector_idx, sector_idx) contains the eigenvectors
for that sector as columns.

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c = get_annihilators(basis)
H = sum(c[i]' * c[i] for i in 1:2)

result = solve_hamiltonian(H)

# Access eigenvalues for sector 2
E_sector2 = result.eigenvalues.sector_diags[2]

# Access eigenvectors for sector 2
U_sector2 = result.eigenvectors.blocks[(2, 2)]
```
"""
struct EigenResult
    eigenvalues::DiagonalOperator
    eigenvectors::Operator
end

"""
    solve_hamiltonian(H::Operator) -> EigenResult

Diagonalize a Hamiltonian efficiently by processing each sector independently.

This function leverages the block-diagonal structure of the Hamiltonian (implied by flux=0)
to massively reduce computational cost compared to full diagonalization.

Arguments:
- `H::Operator`: Hamiltonian (must have flux=0)

Returns:
- `EigenResult`: Structure containing eigenvalues and eigenvectors as operator types.

Requirements:
- H must have flux = 0 (block-diagonal in the chosen basis)
- H should be Hermitian

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
c = get_annihilators(basis)
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
function solve_hamiltonian(H::Operator)
    basis = H.basis

    # Get concrete quantum number type from first element
    QN_type = typeof(basis.site_qns[1])

    # Verify flux is zero (Hamiltonian is block-diagonal)
    if H.flux.delta_qn != zero_qn(QN_type)
        error("Hamiltonian must have zero flux for efficient block diagonalization. " *
              "Got flux = $(H.flux.delta_qn)")
    end

    # Storage for eigenvalues and eigenvectors
    sector_diags = Dict{Int, Vector{Float64}}()
    evec_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    # Diagonalize each sector independently
    for (sector_idx, sector) in enumerate(basis.sectors)
        dim = sector.dim

        # Extract block for this sector (diagonal block)
        if haskey(H.blocks, (sector_idx, sector_idx))
            block = H.blocks[(sector_idx, sector_idx)]

            # Diagonalize this block
            # Make sure it's Hermitian
            sol = eigen(Hermitian(block))

            sector_diags[sector_idx] = sol.values
            evec_blocks[(sector_idx, sector_idx)] = sol.vectors
        else
            # No block means this sector has zero energy
            # (e.g., vacuum sector for interaction-only Hamiltonian)
            sector_diags[sector_idx] = zeros(Float64, dim)
            evec_blocks[(sector_idx, sector_idx)] = Matrix{ComplexF64}(I, dim, dim)
        end
    end

    # Create operator types
    zero_flux = OperatorFlux(zero_qn(QN_type))
    eigenvalues = DiagonalOperator(basis, zero_flux, sector_diags)
    eigenvectors = Operator(basis, zero_flux, evec_blocks)

    return EigenResult(eigenvalues, eigenvectors)
end
