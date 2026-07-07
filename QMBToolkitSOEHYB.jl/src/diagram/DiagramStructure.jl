# Diagram backbone structure
#
# BackBone of M nodes and M-1 edges:
#   |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
#
# Accompanies L decomposed hybridization functions:
#   - Propagator diagram of bare expansion: L = (M-2)/2
#   - Self-energy diagram of bold expansion: L = (M-2)/2 (one undecomposed hyb line to 0)
#   - Green's function diagram: L = (M-3)/2
# Each of the L decomposed hybridization functions can be in p states.

"""
    HybridizationState

Compact representation of hybridization state using integer encoding:

  - index == 0: NoHyb (no hybridization)
  - index > 0: KPlus with l = index
  - index < 0: KMinus with l = -index

This concrete type representation provides optimal performance compared to
abstract type hierarchies, with zero dispatch overhead and efficient memory layout.
"""
struct HybridizationState
    index::Int  # 0 for NoHyb, positive for KPlus(l), negative for KMinus(l)
end

# Constructor helpers for clarity
NoHyb() = HybridizationState(0)
KPlus(l::Int) = HybridizationState(l)
KMinus(l::Int) = HybridizationState(-l)

# Type checking helpers
is_nohyb(h::HybridizationState) = h.index == 0
is_kplus(h::HybridizationState) = h.index > 0
is_kminus(h::HybridizationState) = h.index < 0
get_kplus_index(h::HybridizationState) = h.index
get_kminus_index(h::HybridizationState) = -h.index

struct BackBoneNode
    opt_type::Symbol # :c or :cdag or :empty
    flavor::Int # spin-orbital index associated with the creation or annihilation operator; -1 for :empty nodes
    bar_index::Int # can take 1 to p if the operator has bar, otherwise -1
    attached_hyb::HybridizationState # can be NoHyb, KPlus(l), or KMinus(l)
end

#=============================================================================
# Mutable BackBone for Zero-Allocation Iteration
=============================================================================#

"""
    MutableBackBoneEdge

Mutable edge structure for zero-allocation diagram iteration.
The attached_hybs vector is pre-allocated and reused via clear!/push!.

# Fields

  - `attached_hybs::Vector{HybridizationState}`: Pre-allocated vector, cleared between uses
  - `_max_hybs::Int`: Maximum capacity (for bounds checking)
"""
mutable struct MutableBackBoneEdge
    attached_hybs::Vector{HybridizationState}
    _length::Int  # Current logical length (avoids resize! overhead)
end

MutableBackBoneEdge(max_hybs::Int) = MutableBackBoneEdge(
    Vector{HybridizationState}(undef, max_hybs), 0)

"""
    MutableBackBone

Mutable backbone structure for zero-allocation diagram iteration.
Pre-allocates all arrays and reuses them between diagram generations.

Use `reset_backbone!` to clear state and `fill_backbone_from_configs!` to populate.

# Fields

  - `diagram_type::Symbol`: :propagator_diagram, :selfenergy_diagram, :green_diagram
  - `sign::Float64`: Overall sign factor (mutable)
  - `prefactor_Kexps::Vector{Int}`: Exponent counts (pre-allocated, filled with zeros)
  - `nodes::Vector{BackBoneNode}`: Node array (pre-allocated)
  - `edges::Vector{MutableBackBoneEdge}`: Edge array (pre-allocated with capacity)
  - `external_opts_loc::Vector{Int}`: External operator locations (pre-allocated)
  - `M::Int`: Current number of nodes in use
"""
mutable struct MutableBackBone
    diagram_type::Symbol
    sign::Float64
    prefactor_Kexps::Vector{Int}
    nodes::Vector{BackBoneNode}
    edges::Vector{MutableBackBoneEdge}
    external_opts_loc::Vector{Int}
    M::Int  # Current number of nodes (may be less than allocated)
end

"""
    MutableBackBone(max_M::Int, p::Int, max_hybs_per_edge::Int)

Create a pre-allocated MutableBackBone with given capacities.

# Arguments

  - `max_M::Int`: Maximum number of nodes (determines edge count = max_M - 1)
  - `p::Int`: Number of SOE exponents
  - `max_hybs_per_edge::Int`: Maximum hybridizations per edge

# Returns

  - `MutableBackBone`: Pre-allocated backbone ready for use
"""
function MutableBackBone(max_M::Int, p::Int, max_hybs_per_edge::Int)
    empty_node = BackBoneNode(:empty, -1, -1, NoHyb())
    nodes = fill(empty_node, max_M)
    edges = [MutableBackBoneEdge(max_hybs_per_edge) for _ in 1:(max_M - 1)]
    prefactor_Kexps = zeros(Int, p)
    external_opts_loc = Vector{Int}(undef, 2)  # Max 2 external operators
    return MutableBackBone(:propagator_diagram, 1.0, prefactor_Kexps, nodes, edges, external_opts_loc, 2)
end

"""
    reset_backbone!(bb::MutableBackBone)

Reset a MutableBackBone for reuse. Clears all state without deallocating.
"""
function reset_backbone!(bb::MutableBackBone)
    bb.sign = 1.0
    fill!(bb.prefactor_Kexps, 0)
    # Clear edge hybridizations (just reset length counter)
    for edge in bb.edges
        edge._length = 0
    end
    # Nodes will be overwritten during fill
    return bb
end

# Helper to push to MutableBackBoneEdge without allocation
function push_hyb!(edge::MutableBackBoneEdge, hyb::HybridizationState)
    edge._length += 1
    @boundscheck @assert edge._length <= length(edge.attached_hybs) "Edge hybridization overflow"
    @inbounds edge.attached_hybs[edge._length] = hyb
end

# View of active hybridizations (for iteration during evaluation)
function active_hybs(edge::MutableBackBoneEdge)
    return view(edge.attached_hybs, 1:edge._length)
end

# Accessors for MutableBackBone
get_attached_hybs(edge::MutableBackBoneEdge) = active_hybs(edge)
get_external_opts(bb::MutableBackBone) = view(bb.external_opts_loc, 1:2)
