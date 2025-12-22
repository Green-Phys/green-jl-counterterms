"""
Diagonal operators for block-dense Hilbert space representation.

Diagonal operators store only the diagonal elements per sector, providing efficient
storage for eigenvalues and other block-diagonal matrices.
"""

"""
    BlockDenseDiagonalOperator

A block-diagonal operator storing only diagonal elements per sector.

Fields:
- `basis::BlockDenseBasis`: The basis this operator acts on
- `flux::OperatorFlux`: Quantum number change (must be zero for diagonal operators)
- `sector_diags::Dict{Int, Vector{Float64}}`: Diagonal elements per sector

This type is used to efficiently store eigenvalues from block-diagonal Hamiltonian
diagonalization. The flux must always be zero since diagonal operators only connect
states within the same sector.

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
H = ...  # Build Hamiltonian
result = solve_hamiltonian_blockdense(H)

# Access eigenvalues for sector 2
evals_sector2 = result.eigenvalues.sector_diags[2]
```
"""
struct BlockDenseDiagonalOperator
    basis::BlockDenseBasis
    flux::OperatorFlux
    sector_diags::Dict{Int, Vector{Float64}}

    function BlockDenseDiagonalOperator(
        basis::BlockDenseBasis,
        flux::OperatorFlux,
        sector_diags::Dict{Int, Vector{Float64}}
    )
        # Validate flux is zero
        QN_type = typeof(basis.site_qns[1])
        if flux.delta_qn != zero_qn(QN_type)
            error("BlockDenseDiagonalOperator must have zero flux. Got: $(flux.delta_qn)")
        end

        # Validate sector dimensions
        for (sector_idx, diag) in sector_diags
            @assert 1 <= sector_idx <= length(basis.sectors) "Invalid sector index: $sector_idx"
            expected_dim = basis.sectors[sector_idx].dim
            @assert length(diag) == expected_dim "Diagonal length ($(length(diag))) must match sector dimension ($expected_dim) for sector $sector_idx"
        end

        new(basis, flux, sector_diags)
    end
end

# Convenience constructor that infers zero flux from basis
function BlockDenseDiagonalOperator(
    basis::BlockDenseBasis,
    sector_diags::Dict{Int, Vector{Float64}}
)
    QN_type = typeof(basis.site_qns[1])
    flux = OperatorFlux(zero_qn(QN_type))
    BlockDenseDiagonalOperator(basis, flux, sector_diags)
end

# Allow construction from any real vectors
function BlockDenseDiagonalOperator(
    basis::BlockDenseBasis,
    sector_diags::Dict{Int, Vector{<:Real}}
)
    converted = Dict{Int, Vector{Float64}}(
        k => convert(Vector{Float64}, v) for (k, v) in sector_diags
    )
    BlockDenseDiagonalOperator(basis, converted)
end

"""
    Base.getindex(D::BlockDenseDiagonalOperator, sector_idx::Int, i::Int)

Get the i-th diagonal element in the specified sector.
"""
function Base.getindex(D::BlockDenseDiagonalOperator, sector_idx::Int, i::Int)
    return D.sector_diags[sector_idx][i]
end

"""
    Base.getindex(D::BlockDenseDiagonalOperator, sector_idx::Int)

Get the full diagonal vector for the specified sector.
"""
function Base.getindex(D::BlockDenseDiagonalOperator, sector_idx::Int)
    return D.sector_diags[sector_idx]
end

"""
    Base.haskey(D::BlockDenseDiagonalOperator, sector_idx::Int)

Check if the diagonal has values for the specified sector.
"""
Base.haskey(D::BlockDenseDiagonalOperator, sector_idx::Int) = haskey(D.sector_diags, sector_idx)

# Display
function Base.show(io::IO, op::BlockDenseDiagonalOperator)
    QN_type = eltype(op.basis.site_qns)
    println(io, "BlockDenseDiagonalOperator{$QN_type}")
    println(io, "  flux: ", op.flux.delta_qn)
    println(io, "  num_sectors: ", length(op.sector_diags))

    total_elements = sum(length(diag) for diag in values(op.sector_diags))
    println(io, "  total_elements: ", total_elements)
end
