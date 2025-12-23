"""
Representation format traits for operator and propagator storage.

Two formats supported:
- DenseFormat: Standard dense matrix representation (current default)
- BlockDenseFormat: Block-dense representation exploiting quantum number conservation
"""

abstract type HilbertSpaceFormat end

"""
    DenseFormat <: HilbertSpaceFormat

Dense matrix representation using full Hilbert space.
Operators are Matrix{T}, propagators are Array{T,3}.
"""
struct DenseFormat <: HilbertSpaceFormat end

"""
    BlockDenseFormat <: HilbertSpaceFormat

Block-dense representation exploiting quantum number conservation.
Operators are BlockDenseOperator, propagators are Vector{BlockDenseOperator}.
"""
struct BlockDenseFormat <: HilbertSpaceFormat end

Base.show(io::IO, ::DenseFormat) = print(io, "DenseFormat")
Base.show(io::IO, ::BlockDenseFormat) = print(io, "BlockDenseFormat")

"""
    ed_method(::Type{<:HilbertSpaceFormat}) -> Symbol

Return the QMBToolkitED method symbol for Hamiltonian construction.
"""
ed_method(::Type{DenseFormat}) = :dense
ed_method(::Type{BlockDenseFormat}) = :blockdense
