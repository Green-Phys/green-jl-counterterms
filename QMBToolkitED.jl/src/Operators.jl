"""
Operator types for quantum many-body systems.

Operators are stored as collections of dense matrix blocks, with each block
connecting two symmetry sectors. The flux (quantum number change) determines
which sector pairs are connected.
"""

# ============================================================================
# Operator Type
# ============================================================================

"""
    Operator{T}

A block-dense matrix operator exploiting quantum number conservation.

Type Parameters:
- `T<:Number`: Element type (Float64 for real operators, ComplexF64 for complex)

Fields:
- `basis::Basis`: The basis this operator acts on
- `flux::QuantumNumber`: Quantum number change carried by this operator
- `blocks::Dict{Tuple{Int,Int}, Matrix{T}}`: Non-zero blocks

The blocks are indexed as `(target_sector, source_sector) → Matrix`.
Matrix element `block[i,j]` maps state j in source sector to state i in target sector.

This follows the standard linear algebra convention: operator O has matrix element
`O[α, β]` connecting state β (source) to state α (target).

Example:
```julia
# Create annihilation operator
basis = Basis([QN_N(1), QN_N(1)])
c1 = annihilator(basis, 1)

# Check flux
println(c1.flux)  # QN_N(-1)

# Check blocks
for ((target, source), block) in c1.blocks
    println("Block (\$target ← \$source): ", size(block))
end
```
"""
struct Operator{T<:Number}
    basis::Basis
    flux::QuantumNumber
    blocks::Dict{Tuple{Int,Int},Matrix{T}}
end

# Display
function Base.show(io::IO, op::Operator{T}) where {T}
    QN_type = eltype(op.basis.flavor_qns)
    println(io, "Operator{$T, $QN_type}")
    println(io, "  flux: ", op.flux)
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
- `flux::QuantumNumber`: Quantum number change (must be zero for diagonal operators)
- `sector_diags::Dict{Int, Vector{Float64}}`: Diagonal elements per sector

This type is used to efficiently store eigenvalues from block-diagonal Hamiltonian
diagonalization. The flux must always be zero since diagonal operators only connect
states within the same sector.

Example:
```julia
basis = Basis([QN_N(1), QN_N(1)])
H = ...  # Build Hamiltonian
result = solve_hamiltonian(H)

# Access eigenvalues for sector 2
evals_sector2 = result.eigenvalues.sector_diags[2]
```
"""
struct DiagonalOperator
    basis::Basis
    flux::QuantumNumber
    sector_diags::Dict{Int,Vector{Float64}}

    function DiagonalOperator(
        basis::Basis,
        flux::QuantumNumber,
        sector_diags::Dict{Int,Vector{Float64}},
    )
        # Validate flux is zero
        QN_type = typeof(basis.flavor_qns[1])
        if flux != zero(QN_type)
            error("DiagonalOperator must have zero flux. Got: $(flux)")
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

# Display
function Base.show(io::IO, op::DiagonalOperator)
    QN_type = eltype(op.basis.flavor_qns)
    println(io, "DiagonalOperator{$QN_type}")
    println(io, "  flux: ", op.flux)
    println(io, "  num_sectors: ", length(op.sector_diags))

    total_elements = sum(length(diag) for diag in values(op.sector_diags))
    println(io, "  total_elements: ", total_elements)
end

# ============================================================================
# Annihilator Construction
# ============================================================================

"""
    annihilator(basis::Basis, flavor::Int) -> Operator

Construct a fermionic annihilation operator for flavor `flavor`.

Arguments:
- `basis::Basis`: The many-body basis
- `flavor::Int`: 1-indexed flavor index to annihilate

Returns:
- `Operator`: The annihilation operator c_{flavor}

The operator has flux `-basis.flavor_qns[flavor]`, which means:
- Annihilating flavor k removes the quantum number flavor_qns[k]
- Only sectors with compatible quantum numbers are connected

Algorithm:
1. Determine flux from flavor_qns[flavor]
2. For each source sector:
   a. Compute target sector from source_qn + flux
   b. If target sector doesn't exist, skip
   c. Allocate block matrix (target.dim × source.dim)
   d. For each source state:
      - Check if flavor is occupied (bit set)
      - Compute target state (flip bit)
      - Compute fermionic sign
      - Set matrix element
3. Store non-zero blocks

Example:
```julia
# 2-flavor spin system
flavor_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]  # [↑, ↓]
basis = Basis(flavor_qns)

# Annihilate spin-up electron
c_up = annihilator(basis, 1)
# flux = QN_NSz(-1, -1)

# Annihilate spin-down electron
c_down = annihilator(basis, 2)
# flux = QN_NSz(-1, 1)
```
"""
function annihilator(basis::Basis, flavor::Int)
    @assert 1 <= flavor <= basis.nflavors "flavor must be in 1:nflavors"

    # Determine flux: annihilating flavor k removes flavor_qns[k]
    flux = -basis.flavor_qns[flavor]

    # Bit mask for this flavor
    bitmask = 1 << (flavor - 1)

    # Storage for blocks - use Float64 since annihilator values are ±1
    blocks = Dict{Tuple{Int,Int},Matrix{Float64}}()

    # Iterate over source sectors
    for (src_idx, src_sector) in enumerate(basis.sectors)
        # Compute target quantum number
        target_qn = src_sector.qn + flux

        # Check if target sector exists
        if !haskey(basis.qn_to_sector, target_qn)
            continue  # No connection to this target
        end

        tgt_idx = basis.qn_to_sector[target_qn]
        tgt_sector = basis.sectors[tgt_idx]

        # Allocate block: rows = target states, cols = source states
        # Use Float64 since fermion_sign returns ±1
        block = zeros(Float64, tgt_sector.dim, src_sector.dim)

        # Fill block by iterating over source states
        for (src_local_idx, src_state) in enumerate(src_sector.states)
            # Check if flavor is occupied
            if (src_state & bitmask) != 0
                # Flavor is occupied, can be annihilated

                # Apply annihilation: flip the bit
                tgt_state = src_state ⊻ bitmask  # XOR to flip bit

                # Compute fermionic sign
                sign = fermion_sign(src_state, flavor)

                # Find local index of target state in target sector
                tgt_local_idx = tgt_sector.state_to_index[tgt_state]

                # Set matrix element
                block[tgt_local_idx, src_local_idx] = sign
            end
        end

        # Block is guaranteed non-zero: the sector connection exists only when
        # at least one source state has the flavor occupied
        blocks[(tgt_idx, src_idx)] = block
    end

    return Operator(basis, flux, blocks)
end

"""
    annihilators(basis::Basis) -> Vector{Operator}

Generate fermionic annihilation operators for a given basis.

Arguments:
- `basis::Basis`: The many-body basis with sector structure

Returns:
- `Vector{Operator}`: Array of annihilation operators c[i] for i=1:nflavors

Example:
```julia
# Create basis for 3-flavor system with N conservation
flavor_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = Basis(flavor_qns)

# Get annihilation operators
c = annihilators(basis)

# c[i] is the annihilation operator for flavor i
```
"""
function annihilators(basis::Basis)
    return [annihilator(basis, i) for i = 1:basis.nflavors]
end
