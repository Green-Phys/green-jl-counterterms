"""
Block-dense basis construction for quantum many-body systems.

The basis is constructed from a site quantum number list, which specifies
the quantum numbers carried by each single-particle orbital. States are
then grouped into sectors by their total quantum number.
"""

"""
    BlockDenseBasis

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
basis = make_basis_blockdense(site_qns)
println("Number of sectors: ", length(basis.sectors))
println("Total dimension: ", basis.total_dim)
```
"""
struct BlockDenseBasis
    site_qns::Vector{QuantumNumber}
    nflavors::Int
    sectors::Vector{Sector}
    qn_to_sector::Dict{QuantumNumber,Int}
    total_dim::Int

    # Inner constructor with validation
    function BlockDenseBasis(
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

"""
    make_basis_blockdense(site_qns::Vector{QN}) where {QN <: QuantumNumber} -> BlockDenseBasis

Construct a block-dense basis from site quantum numbers.

This is the primary interface for creating a basis. Each element of `site_qns`
specifies the quantum number carried by occupying that orbital.

Arguments:
- `site_qns::Vector{QN}`: Quantum numbers for each single-particle orbital

Returns:
- `BlockDenseBasis`: The partitioned many-body basis

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
basis = make_basis_blockdense(site_qns)

# Should have 4 sectors:
# Sector 1: N=0, Sz=0  (vacuum)
# Sector 2: N=1, Sz=-1 (spin down)
# Sector 3: N=1, Sz=+1 (spin up)
# Sector 4: N=2, Sz=0  (both occupied)
```

Example with N-only conservation:
```julia
site_qns = [QN_N(1), QN_N(1), QN_N(1)]
basis = make_basis_blockdense(site_qns)

# Should have 4 sectors: N=0, 1, 2, 3
# With dimensions: 1, 3, 3, 1 (binomial coefficients)
```
"""
function make_basis_blockdense(site_qns::Vector{QN}) where {QN <: QuantumNumber}
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
    return BlockDenseBasis(site_qns, nflavors, sectors, qn_to_sector, nfock)
end

# Display
function Base.show(io::IO, basis::BlockDenseBasis)
    QN_type = eltype(basis.site_qns)
    println(io, "BlockDenseBasis{$QN_type}")
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
