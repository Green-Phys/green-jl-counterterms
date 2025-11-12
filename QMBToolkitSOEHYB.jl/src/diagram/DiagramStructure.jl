# Normal diagrams structure

# Convert a normal diagram to a list of BackBone diagrams

# BackBone structure

"""
    BackBone of M nodes and M-1 edges:
    |Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
    We assume it accompanies of L decomposed hbyridization functions:
    - Propagator diagram of the bare expansion: L = (M-2)/2
    - Self-energy diagram of the bold expansion: L = (M-2)/2 # there is one undecomposed hybrdization line connected to 0
    - Green's function diagram: L = (M-3)/2
    We assme each of the L decomposed hybrdization functions can be in r states
"""

abstract type HybridizationState end
struct NoHyb <: HybridizationState end
mutable struct DeltaHyb <: HybridizationState 
    i::Int # flavor index for Delta_{ij} associated with a c^{\dag}_{i}
    j::Int # flavor index for Delta_{ij} associated with a c_{j}
end
mutable struct KPlus <: HybridizationState
    l::Int # index of the decomposed hybridization function, runs from 1 to L
end
mutable struct KMinus <: HybridizationState
    l::Int # index of the decomposed hybridization function, runs from 1 to L
end

mutable struct BackBoneNode
    opt_type::Symbol # :c or :cdag or :empty
    flavor::Union{Int, Nothing} # spin-orbital index associated with the creation or annihilation operator
    bar_index::Union{Int, Nothing} # can take 1 to r if the operator has bar, otherwise nothing
    attached_hyb::HybridizationState # can be NoHyb, DeltaHyb, KPlus(l), KMinus(l)
end


mutable struct BackBoneEdge
    attached_hybs::Vector{HybridizationState} # length <=L, each can be KPlus(l) or KMinus(l) with distinct l
end

struct BackBone 
    prefactor::Float64 # overall prefactor = sign * 1/[(K_{1}^{\Ksign_{1}})^{Kexp_{1}}...(K_{L}^{K_sign{L}})^{Kexp_{L}}]
    sign::Float64
    prefactor_Ksigns::Vector{Int} 
    prefactor_Kexps::Vector{Int}  
    nodes::Vector{BackBoneNode} # length M
    edges::Vector{BackBoneEdge} # length M-1
    exponentials_index::Vector{Int} # length L, index of the exponentials in the decomposed hybridization functions, runs from 1 to r
end

