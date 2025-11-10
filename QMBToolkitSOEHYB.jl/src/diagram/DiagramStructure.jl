# Normal diagrams structure

# Convert a normal diagram to a list of skeleton diagrams

# Skeleton structure

struct SkeletonNode
    time_index::UInt # index of the time variable
    flavor::UInt # spin-orbital index
    is_creation::Bool # true for creation, false for annihilation
    attached_hyb::HybridizationState # can be NoHyb, DeltaHyb, KPlus(l), KMinus(l)
end

struct SkeletonEdge
    time_start_index::UInt
    time_end_index::UInt
    KPlus_list::Vector{KPlus} # list of KPlus hybridizations along this edge
end

struct Skeleton
    n_taus::UInt # at least 2 which are the first index for 0 and the last index for tau or beta
    prefactor::Float64
    nodes::Vector{SkeletonNode}
    edges::Vector{SkeletonEdge}
end



abstract type HybridizationState end
struct NoHyb <: HybridizationState end
struct DeltaHyb <: HybridizationState end
struct KPlus <: HybridizationState
    l::Int
end
struct KMinus <: HybridizationState
    l::Int
end
