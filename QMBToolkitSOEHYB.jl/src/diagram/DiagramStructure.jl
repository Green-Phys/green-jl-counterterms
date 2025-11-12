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
struct BackBone 
    prefactor::Float64 # overall prefactor = sign * 1/[(K_{1}^{\Ksign_{1}})^{Kexp_{1}}...(K_{L}^{K_sign{L}})^{Kexp_{L}}]
    prefactor_Ksigns::Vector{int} 
    prefactor_Kexps::Vector{int}  
    nodes::Vector{BackBoneNode} # length M
    edges::Vector{BackBoneEdge} # length M-1
    exponentials_index::Vector{Int} # length L, index of the exponentials in the decomposed hybridization functions, runs from 1 to r
end

mutable struct BackBoneNode
    is_creation::Bool # true for creation, false for annihilation
    flavor::Int # spin-orbital index associated with the creation or annihilation operator
    has_bar::Bool # true if there is a bar on the creation or annihilation operator
    attached_hyb::HybridizationState # can be NoHyb, DeltaHyb, KPlus(l), KMinus(l)
end


mutable struct BackBoneEdge
    has_propagator::Bool # true if there is a propagator line on this edge
    attached_hybs::Vector{HybridizationState} # length <=L, each can be KPlus(l) or KMinus(l) with distinct l
end

abstract type HybridizationState end
struct NoHyb <: HybridizationState end
struct DeltaHyb <: HybridizationState end
struct KPlus <: HybridizationState
    l::Int # index of the decomposed hybridization function, runs from 1 to L
end
struct KMinus <: HybridizationState
    l::Int # index of the decomposed hybridization function, runs from 1 to L
end
