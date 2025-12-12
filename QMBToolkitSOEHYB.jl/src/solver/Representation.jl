"""
Representation format traits for operator and propagator storage.

Two formats supported:
- DenseFormat: Standard dense matrix representation (current default)
- BlockDenseFormat: Block-sparse representation exploiting quantum number conservation
"""

abstract type RepresentationFormat end

"""
    DenseFormat <: RepresentationFormat

Dense matrix representation using full Hilbert space.
Operators are Matrix{T}, propagators are Array{T,3}.
"""
struct DenseFormat <: RepresentationFormat end

"""
    BlockDenseFormat <: RepresentationFormat

Block-dense representation exploiting quantum number conservation.
Operators are BlockDenseOperator, propagators are Vector{BlockDenseOperator}.
"""
struct BlockDenseFormat <: RepresentationFormat end

# Trait dispatch helpers
representation_format(::Val{:dense}) = DenseFormat()
representation_format(::Val{:blockdense}) = BlockDenseFormat()

# String conversion for I/O
format_symbol(::DenseFormat) = :dense
format_symbol(::BlockDenseFormat) = :blockdense

# Display
Base.show(io::IO, ::DenseFormat) = print(io, "DenseFormat")
Base.show(io::IO, ::BlockDenseFormat) = print(io, "BlockDenseFormat")
