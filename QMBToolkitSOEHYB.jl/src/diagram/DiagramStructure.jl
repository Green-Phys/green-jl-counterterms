# Normal diagrams structure

# Convert a normal diagram to a list of BackBone diagrams

# BackBone structure

"""
    BackBone of M nodes and M-1 edges:
    |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    We assume it accompanies of L decomposed hybridization functions:
    - Propagator diagram of the bare expansion: L = (M-2)/2
    - Self-energy diagram of the bold expansion: L = (M-2)/2 # there is one undecomposed hybridization line connected to 0
    - Green's function diagram: L = (M-3)/2
    We assume each of the L decomposed hybridization functions can be in p states
"""

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


struct BackBoneEdge
    attached_hybs::Vector{HybridizationState} # length <=L, each can be KPlus(l) or KMinus(l)
end

struct BackBone
    diagram_type::Symbol # :propagator_diagram, :selfenergy_diagram, :green_diagram
    sign::Float64 # the overall prefactor is sign * 1/(K_{1}^{sign_{1}}(0))^{Kexp_{1}} * ... * 1/(K_{p}^{sign_{p}}(0))^{Kexp_{p}}; sign_{i} is - if K_exp{i} <=0 else +
    prefactor_Kexps::Vector{Int} # Length p
    nodes::Vector{BackBoneNode} # length M
    edges::Vector{BackBoneEdge} # length M-1
    external_opts_loc::Vector{Int} # only for green_diagram, record the location of external operators (annihilation and creation); empty vector for other diagram types
end

