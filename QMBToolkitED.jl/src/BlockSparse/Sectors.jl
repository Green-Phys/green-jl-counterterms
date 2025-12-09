"""
Sector type for grouping Fock states by quantum numbers.

A Sector contains all many-body states that share the same total quantum number.
"""

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
