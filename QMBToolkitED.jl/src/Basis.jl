"""
Sector and Basis types for quantum many-body systems.

A Sector contains all many-body states that share the same total quantum number.
A Basis is a collection of sectors that partition the full Hilbert space.
"""

# ============================================================================
# Sector Type
# ============================================================================

"""
    Sector

A symmetry sector containing all Fock states with the same quantum number.

Fields:
- `qn::QuantumNumber`: The quantum number characterizing this sector
- `states::Vector{Int}`: Bit-pattern integers representing Fock states
- `state_to_index::Dict{Int,Int}`: Maps Fock state → local index within sector
- `dim::Int`: Number of states in this sector (== length(states))

The states are stored as bit-pattern integers (representing 0-indexed occupation numbers)
in 1-indexed Julia arrays. For example, with nflavors=3:
- State |000⟩ (vacuum) → integer 0
- State |001⟩ (orbital 0 occupied) → integer 1
- State |101⟩ (orbitals 0 and 2 occupied) → integer 5

Example:
```julia
# Create a sector manually
states = [0, 1, 2, 4]  # States with N=0 or N=1
state_to_index = Dict(0=>1, 1=>2, 2=>3, 4=>4)
qn = QN_N(1)
sector = Sector(qn, states, state_to_index, 4)

# Lookup
idx = sector.state_to_index[1]  # Get local index of state |001⟩
state = sector.states[idx]       # Get state from index
```
"""
struct Sector
    qn::QuantumNumber
    states::Vector{Int}
    state_to_index::Dict{Int,Int}
    dim::Int

    # Inner constructor with validation
    function Sector(qn::QuantumNumber, states::Vector{Int}, state_to_index::Dict{Int,Int}, dim::Int)
        @assert length(states) == dim "Dimension mismatch: length(states)=$(length(states)) != dim=$dim"
        @assert length(state_to_index) == dim "Dimension mismatch: length(state_to_index)=$(length(state_to_index)) != dim=$dim"
        new(qn, states, state_to_index, dim)
    end
end

"""
    Sector(qn::QuantumNumber, states::Vector{Int})

Convenience constructor that builds state_to_index and dim automatically.

Example:
```julia
states = [0, 1, 2, 4]
qn = QN_N(1)
sector = Sector(qn, states)
```
"""
function Sector(qn::QuantumNumber, states::Vector{Int})
    state_to_index = Dict(state => idx for (idx, state) in enumerate(states))
    dim = length(states)
    return Sector(qn, states, state_to_index, dim)
end

# Display
function Base.show(io::IO, sector::Sector)
    print(io, "Sector(qn=", sector.qn, ", dim=", sector.dim, ")")
end

# ============================================================================
# Basis Type
# ============================================================================

"""
    Basis

A many-body basis partitioned into symmetry sectors.

Fields:
- `site_qns::Vector{QuantumNumber}`: Quantum number for each orbital/flavor
- `nflavors::Int`: Number of orbitals (== length(site_qns))
- `sectors::Vector{Sector}`: All symmetry sectors, sorted by quantum number
- `qn_to_sector::Dict{QuantumNumber,Int}`: Maps quantum number → sector index
- `total_dim::Int`: Total Hilbert space dimension (== 2^nflavors)

Example:
```julia
# 4-flavor Hubbard atom with (N, Sz) conservation
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1), QN_NSz(1, 1), QN_NSz(1, -1)]
basis = make_basis(site_qns)
println("Number of sectors: ", length(basis.sectors))
println("Total dimension: ", basis.total_dim)
```
"""
struct Basis
    site_qns::Vector{QuantumNumber}
    nflavors::Int
    sectors::Vector{Sector}
    qn_to_sector::Dict{QuantumNumber,Int}
    total_dim::Int

    # Inner constructor with validation
    function Basis(
        site_qns::Vector{QN},
        nflavors::Int,
        sectors::Vector{Sector},
        qn_to_sector::Dict{QuantumNumber,Int},
        total_dim::Int
    ) where {QN <: QuantumNumber}
        @assert length(site_qns) == nflavors "site_qns length must equal nflavors"
        @assert total_dim == 1 << nflavors "total_dim must equal 2^nflavors"

        # Validate total dimension matches sum of sector dimensions
        sum_dim = sum(s.dim for s in sectors)
        @assert sum_dim == total_dim "Sum of sector dimensions ($sum_dim) != total_dim ($total_dim)"

        # Validate all sectors are in the map
        @assert length(qn_to_sector) == length(sectors) "qn_to_sector size mismatch"

        new(site_qns, nflavors, sectors, qn_to_sector, total_dim)
    end
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    sum_occupied_sites(state::Int, site_qns::Vector{QN}) -> QN where QN <: QuantumNumber

Compute the total quantum number of a many-body state by summing the quantum numbers
of all occupied single-particle sites.

Arguments:
- `state::Int`: Bit-pattern integer (0-indexed) representing the Fock state
- `site_qns::Vector{QN}`: Quantum number for each single-particle orbital

Returns:
- Total quantum number obtained by summing site_qns[i+1] for all occupied bits i

The state is interpreted as a bit pattern where bit i (0-indexed) corresponds to
the occupation of orbital i. Julia arrays are 1-indexed, so bit i maps to site_qns[i+1].

Example:
```julia
# 2-site spin system: [↑, ↓]
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]

# State |10⟩ (binary) = 2 (decimal): bit 1 set → orbital 1 occupied (0-indexed)
# This corresponds to site_qns[2] (Julia's 1-indexed array), which is the ↓ spin
state = 2
total_qn = sum_occupied_sites(state, site_qns)  # QN_NSz(1, -1)

# State |11⟩ (binary) = 3 (decimal): both orbitals occupied
state = 3
total_qn = sum_occupied_sites(state, site_qns)  # QN_NSz(2, 0)
```
"""
function sum_occupied_sites(state::Int, site_qns::Vector{QN}) where {QN <: QuantumNumber}
    total = zero_qn(QN)

    for bit in 0:(length(site_qns) - 1)
        # Check if bit is set (orbital occupied)
        if (state >> bit) & 1 == 1
            # Add quantum number of this site (Julia 1-indexed)
            total = combine_qn(total, site_qns[bit + 1])
        end
    end

    return total
end

"""
    compute_fermion_sign(state::Int, flavor::Int) -> Int

Compute the fermionic sign for annihilating orbital `flavor` in `state`.

The sign is (-1)^k where k is the number of occupied orbitals to the right
of `flavor` (i.e., in lower bit positions: bits 0 through flavor-2).

Arguments:
- `state::Int`: Bit-pattern integer (0-indexed) representing the Fock state
- `flavor::Int`: 1-indexed orbital index to annihilate

Returns:
- +1 or -1 depending on the parity of occupied orbitals

This implements the Jordan-Wigner string for fermionic anti-commutation relations.
The convention is that operators are ordered left-to-right in increasing orbital index,
so we count occupied orbitals to the right (lower bit positions).

Example:
```julia
# State |1010⟩ (binary) = 10 (decimal): orbitals 1 and 3 occupied (0-indexed)
state = 10

# Annihilate orbital 2 (1-indexed flavor, bit 1 in 0-indexed)
sign = compute_fermion_sign(state, 2)
# Orbitals below bit 1: bit 0 is NOT set → 0 occupied orbitals
# sign = (-1)^0 = +1

# Annihilate orbital 4 (1-indexed flavor, bit 3 in 0-indexed)
sign = compute_fermion_sign(state, 4)
# Orbitals below bit 3: bits 0,1,2 with only bit 1 set → 1 occupied orbital
# sign = (-1)^1 = -1
```
"""
function compute_fermion_sign(state::Int, flavor::Int)
    # Convert 1-indexed flavor to 0-indexed bit position
    bit_position = flavor - 1

    # Create mask for all bits to the right (lower bit positions)
    # For bit_position=3, mask = 0b0111 = 7 (bits 0,1,2)
    mask = (1 << bit_position) - 1

    # Count occupied orbitals in masked region
    num_occupied = count_ones(state & mask)

    # Fermionic sign: (-1)^num_occupied
    return (-1)^num_occupied
end

# ============================================================================
# Basis Construction
# ============================================================================

"""
    make_basis(site_qns::Vector{QN}) where {QN <: QuantumNumber} -> Basis

Construct a basis from site quantum numbers.

This is the primary interface for creating a basis. Each element of `site_qns`
specifies the quantum number carried by occupying that orbital.

Arguments:
- `site_qns::Vector{QN}`: Quantum numbers for each single-particle orbital

Returns:
- `Basis`: The partitioned many-body basis

Algorithm:
1. Generate all 2^nflavors Fock states (as bit patterns)
2. For each state, compute total QN by summing occupied site QNs
3. Group states by their total QN into sectors
4. Sort sectors by quantum number for deterministic ordering
5. Build lookup map from QN to sector index

Example:
```julia
# Simple 2-flavor spin system: [↑, ↓]
site_qns = [QN_NSz(1, 1), QN_NSz(1, -1)]
basis = make_basis(site_qns)

# Should have 4 sectors:
# Sector 1: N=0, Sz=0  (vacuum)
# Sector 2: N=1, Sz=-1 (spin down)
# Sector 3: N=1, Sz=+1 (spin up)
# Sector 4: N=2, Sz=0  (both occupied)
```

Example with N-only conservation:
```julia
site_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = make_basis(site_qns)

# Should have 4 sectors: N=0, 1, 2, 3
# With dimensions: 1, 3, 3, 1 (binomial coefficients)
```
"""
function make_basis(site_qns::Vector{QN}) where {QN <: QuantumNumber}
    nflavors = length(site_qns)
    nfock = 1 << nflavors

    # Step 1: Group states by quantum number
    qn_groups = Dict{QN, Vector{Int}}()

    for state in 0:(nfock - 1)
        # Compute total quantum number for this state
        total_qn = sum_occupied_sites(state, site_qns)

        # Add to appropriate group
        if haskey(qn_groups, total_qn)
            push!(qn_groups[total_qn], state)
        else
            qn_groups[total_qn] = [state]
        end
    end

    # Step 2: Create sectors in sorted order (deterministic)
    sorted_qns = sort(collect(keys(qn_groups)))
    sectors = Sector[]
    qn_to_sector = Dict{QuantumNumber, Int}()

    for (sector_idx, qn) in enumerate(sorted_qns)
        # Get states for this QN and sort for reproducibility
        states = sort(qn_groups[qn])

        # Create sector
        sector = Sector(qn, states)
        push!(sectors, sector)

        # Build QN → sector index map
        qn_to_sector[qn] = sector_idx
    end

    # Step 3: Construct basis
    return Basis(site_qns, nflavors, sectors, qn_to_sector, nfock)
end

# Display
function Base.show(io::IO, basis::Basis)
    QN_type = eltype(basis.site_qns)
    println(io, "Basis{$QN_type}")
    println(io, "  nflavors: ", basis.nflavors)
    println(io, "  total_dim: ", basis.total_dim)
    println(io, "  num_sectors: ", length(basis.sectors))

    # Show a few sectors
    max_show = 5
    for (i, sector) in enumerate(basis.sectors)
        if i > max_show
            println(io, "  ⋮")
            break
        end
        println(io, "  Sector $i: ", sector.qn, " (dim=$(sector.dim))")
    end
end
