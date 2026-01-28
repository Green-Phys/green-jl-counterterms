"""
Operator types for quantum many-body systems.

Operators are stored as collections of dense matrix blocks, with each block
connecting two symmetry sectors. The flux (quantum number change) determines
which sector pairs are connected.
"""

# ============================================================================
# OperatorFlux Type
# ============================================================================

"""
    OperatorFlux

Quantum number change carried by an operator.

Fields:
- `delta_qn::QuantumNumber`: The quantum number change (flux)

For annihilation operators: flux = reverse_qn(site_qns[flavor])
For creation operators: flux = site_qns[flavor]
For products: flux = combine_qn(flux1, flux2)

Example:
```julia
# Annihilation of spin-up electron
site_qn = QN_NSz(1, 1)
flux_annihilation = OperatorFlux(reverse_qn(site_qn))  # QN_NSz(-1, -1)

# Creation
flux_creation = OperatorFlux(site_qn)  # QN_NSz(1, 1)

# Number operator c† * c
flux_number = OperatorFlux(combine_qn(flux_creation.delta_qn, flux_annihilation.delta_qn))
# Result: QN_NSz(0, 0)
```
"""
struct OperatorFlux
    delta_qn::QuantumNumber
end

# Display
Base.show(io::IO, flux::OperatorFlux) = print(io, "OperatorFlux(", flux.delta_qn, ")")

# ============================================================================
# Operator Type
# ============================================================================

"""
    Operator

A block-dense matrix operator exploiting quantum number conservation.

Fields:
- `basis::Basis`: The basis this operator acts on
- `flux::OperatorFlux`: Quantum number change carried by this operator
- `blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}`: Non-zero blocks

The blocks are indexed as `(target_sector, source_sector) → Matrix`.
Matrix element `block[i,j]` maps state j in source sector to state i in target sector.

This follows the standard linear algebra convention: operator O has matrix element
`O[α, β]` connecting state β (source) to state α (target).

Example:
```julia
# Create annihilation operator
basis = make_basis([QN_N(1), QN_N(1)])
c1 = get_annihilator(basis, 1)

# Check flux
println(c1.flux)  # OperatorFlux(QN_N(-1))

# Check blocks
for ((target, source), block) in c1.blocks
    println("Block (\$target ← \$source): ", size(block))
end
```
"""
struct Operator
    basis::Basis
    flux::OperatorFlux
    blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
end

# Display
function Base.show(io::IO, op::Operator)
    QN_type = eltype(op.basis.site_qns)
    println(io, "Operator{$QN_type}")
    println(io, "  flux: ", op.flux.delta_qn)
    println(io, "  num_blocks: ", length(op.blocks))

    total_nnz = sum(length(block) for block in values(op.blocks))
    println(io, "  total_nonzeros: ", total_nnz)
end

# ============================================================================
# DiagonalOperator Type
# ============================================================================

"""
    DiagonalOperator

A block-diagonal operator storing only diagonal elements per sector.

Fields:
- `basis::Basis`: The basis this operator acts on
- `flux::OperatorFlux`: Quantum number change (must be zero for diagonal operators)
- `sector_diags::Dict{Int, Vector{Float64}}`: Diagonal elements per sector

This type is used to efficiently store eigenvalues from block-diagonal Hamiltonian
diagonalization. The flux must always be zero since diagonal operators only connect
states within the same sector.

Example:
```julia
basis = make_basis([QN_N(1), QN_N(1)])
H = ...  # Build Hamiltonian
result = solve_hamiltonian(H)

# Access eigenvalues for sector 2
evals_sector2 = result.eigenvalues.sector_diags[2]
```
"""
struct DiagonalOperator
    basis::Basis
    flux::OperatorFlux
    sector_diags::Dict{Int, Vector{Float64}}

    function DiagonalOperator(
        basis::Basis,
        flux::OperatorFlux,
        sector_diags::Dict{Int, Vector{Float64}}
    )
        # Validate flux is zero
        QN_type = typeof(basis.site_qns[1])
        if flux.delta_qn != zero_qn(QN_type)
            error("DiagonalOperator must have zero flux. Got: $(flux.delta_qn)")
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
function DiagonalOperator(
    basis::Basis,
    sector_diags::Dict{Int, Vector{Float64}}
)
    QN_type = typeof(basis.site_qns[1])
    flux = OperatorFlux(zero_qn(QN_type))
    DiagonalOperator(basis, flux, sector_diags)
end

# Allow construction from any real vectors
function DiagonalOperator(
    basis::Basis,
    sector_diags::Dict{Int, Vector{<:Real}}
)
    converted = Dict{Int, Vector{Float64}}(
        k => convert(Vector{Float64}, v) for (k, v) in sector_diags
    )
    DiagonalOperator(basis, converted)
end

"""
    Base.getindex(D::DiagonalOperator, sector_idx::Int, i::Int)

Get the i-th diagonal element in the specified sector.
"""
function Base.getindex(D::DiagonalOperator, sector_idx::Int, i::Int)
    return D.sector_diags[sector_idx][i]
end

"""
    Base.getindex(D::DiagonalOperator, sector_idx::Int)

Get the full diagonal vector for the specified sector.
"""
function Base.getindex(D::DiagonalOperator, sector_idx::Int)
    return D.sector_diags[sector_idx]
end

"""
    Base.haskey(D::DiagonalOperator, sector_idx::Int)

Check if the diagonal has values for the specified sector.
"""
Base.haskey(D::DiagonalOperator, sector_idx::Int) = haskey(D.sector_diags, sector_idx)

# Display
function Base.show(io::IO, op::DiagonalOperator)
    QN_type = eltype(op.basis.site_qns)
    println(io, "DiagonalOperator{$QN_type}")
    println(io, "  flux: ", op.flux.delta_qn)
    println(io, "  num_sectors: ", length(op.sector_diags))

    total_elements = sum(length(diag) for diag in values(op.sector_diags))
    println(io, "  total_elements: ", total_elements)
end

# ============================================================================
# Annihilator Construction
# ============================================================================

"""
    get_annihilator(basis::Basis, flavor::Int) -> Operator

Construct a fermionic annihilation operator for orbital `flavor`.

Arguments:
- `basis::Basis`: The many-body basis
- `flavor::Int`: 1-indexed orbital index to annihilate

Returns:
- `Operator`: The annihilation operator c_{flavor}

The operator has flux `reverse_qn(basis.site_qns[flavor])`, which means:
- Annihilating orbital k removes the quantum number site_qns[k]
- Only sectors with compatible quantum numbers are connected

Algorithm:
1. Determine flux from site_qns[flavor]
2. For each source sector:
   a. Compute target sector from source_qn + flux
   b. If target sector doesn't exist, skip
   c. Allocate block matrix (target.dim × source.dim)
   d. For each source state:
      - Check if orbital is occupied (bit set)
      - Compute target state (flip bit)
      - Compute fermionic sign
      - Set matrix element
3. Store non-zero blocks

Example:
```julia
# 2-flavor spin system
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]  # [↑, ↓]
basis = make_basis(site_qns)

# Annihilate spin-up electron
c_up = get_annihilator(basis, 1)
# flux = QN_NSz(-1, -1)

# Annihilate spin-down electron
c_down = get_annihilator(basis, 2)
# flux = QN_NSz(-1, 1)
```
"""
function get_annihilator(basis::Basis, flavor::Int)
    @assert 1 <= flavor <= basis.nflavors "flavor must be in 1:nflavors"

    # Determine flux: annihilating orbital k removes site_qns[k]
    flux = OperatorFlux(reverse_qn(basis.site_qns[flavor]))

    # Bit mask for this orbital
    bitmask = 1 << (flavor - 1)

    # Storage for blocks
    blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    # Iterate over source sectors
    for (src_idx, src_sector) in enumerate(basis.sectors)
        # Compute target quantum number
        target_qn = combine_qn(src_sector.qn, flux.delta_qn)

        # Check if target sector exists
        if !haskey(basis.qn_to_sector, target_qn)
            continue  # No connection to this target
        end

        tgt_idx = basis.qn_to_sector[target_qn]
        tgt_sector = basis.sectors[tgt_idx]

        # Allocate block: rows = target states, cols = source states
        block = zeros(ComplexF64, tgt_sector.dim, src_sector.dim)

        # Fill block by iterating over source states
        for (src_local_idx, src_state) in enumerate(src_sector.states)
            # Check if orbital is occupied
            if (src_state & bitmask) != 0
                # Orbital is occupied, can be annihilated

                # Apply annihilation: flip the bit
                tgt_state = src_state ⊻ bitmask  # XOR to flip bit

                # Compute fermionic sign
                sign = compute_fermion_sign(src_state, flavor)

                # Find local index of target state in target sector
                tgt_local_idx = tgt_sector.state_to_index[tgt_state]

                # Set matrix element
                block[tgt_local_idx, src_local_idx] = sign
            end
        end

        # Store block if non-zero (non-zero blocks expected when sector connection exists)
        if !iszero(block)
            blocks[(tgt_idx, src_idx)] = block
        end
    end

    return Operator(basis, flux, blocks)
end

"""
    get_annihilators(basis::Basis) -> Vector{Operator}

Generate fermionic annihilation operators for a given basis.

Arguments:
- `basis::Basis`: The many-body basis with sector structure

Returns:
- `Vector{Operator}`: Array of annihilation operators c[i] for i=1:nflavors

Example:
```julia
# Create basis for 3-orbital system with N conservation
site_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = make_basis(site_qns)

# Get annihilation operators
c = get_annihilators(basis)

# c[i] is the annihilation operator for flavor i
```
"""
function get_annihilators(basis::Basis)
    return [get_annihilator(basis, i) for i in 1:basis.nflavors]
end
