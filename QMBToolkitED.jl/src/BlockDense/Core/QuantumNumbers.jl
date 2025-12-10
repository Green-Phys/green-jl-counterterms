"""
Quantum number types for block-dense exact diagonalization.

This module defines the abstract QuantumNumber type and concrete implementations
for particle number (N) and particle number with spin (N, Sz) conservation.
"""

# ============================================================================
# Abstract Type
# ============================================================================

"""
    QuantumNumber

Abstract type for quantum numbers used to classify symmetry sectors.

All concrete quantum number types must implement:
- `Base.:(==)` - equality comparison
- `Base.hash` - hashing for Dict usage
- `Base.show` - display
- `combine_qn` - addition of quantum numbers
- `reverse_qn` - negation (for operator adjoint)
- `zero_qn` - identity element
"""
abstract type QuantumNumber end

# ============================================================================
# Concrete Types
# ============================================================================

"""
    QN_NSz(n::Int, sz::Int)

Quantum number for systems with particle number (N) and spin z-component (Sz) conservation.

Fields:
- `n::Int`: Particle number
- `sz::Int`: Spin z-component in units of 1/2 (sz=1 means Sz=1/2, i.e., spin-up)

Example:
```julia
qn_up = QN_NSz(1, 1)    # One spin-up electron
qn_down = QN_NSz(1, -1)  # One spin-down electron
qn_total = combine_qn(qn_up, qn_down)  # QN_NSz(2, 0)
```
"""
struct QN_NSz <: QuantumNumber
    n::Int   # Particle number
    sz::Int  # Spin z-component (in units of 1/2)
end

"""
    QN_N(n::Int)

Quantum number for systems with only particle number (N) conservation.

Simpler variant of QN_NSz for non-spin-polarized systems.

Fields:
- `n::Int`: Particle number

Example:
```julia
qn1 = QN_N(1)
qn2 = QN_N(1)
qn_total = combine_qn(qn1, qn2)  # QN_N(2)
```
"""
struct QN_N <: QuantumNumber
    n::Int   # Particle number
end

# ============================================================================
# Interface Implementation: Equality and Hashing
# ============================================================================

Base.:(==)(a::QN_NSz, b::QN_NSz) = (a.n == b.n) && (a.sz == b.sz)
Base.:(==)(a::QN_N, b::QN_N) = (a.n == b.n)

Base.hash(qn::QN_NSz, h::UInt) = hash((qn.n, qn.sz), h)
Base.hash(qn::QN_N, h::UInt) = hash(qn.n, h)

# ============================================================================
# Interface Implementation: Display
# ============================================================================

function Base.show(io::IO, qn::QN_NSz)
    print(io, "QN_NSz(n=", qn.n, ", sz=", qn.sz, ")")
end

function Base.show(io::IO, qn::QN_N)
    print(io, "QN_N(n=", qn.n, ")")
end

# ============================================================================
# Quantum Number Arithmetic
# ============================================================================

"""
    combine_qn(qn1::QN, qn2::QN) -> QN

Combine two quantum numbers (addition for operators in product).

For QN_NSz: adds both particle number and spin components.
For QN_N: adds particle numbers.

Example:
```julia
qn1 = QN_NSz(1, 1)
qn2 = QN_NSz(1, -1)
combine_qn(qn1, qn2)  # QN_NSz(2, 0)
```
"""
combine_qn(qn1::QN_NSz, qn2::QN_NSz) = QN_NSz(qn1.n + qn2.n, qn1.sz + qn2.sz)
combine_qn(qn1::QN_N, qn2::QN_N) = QN_N(qn1.n + qn2.n)

"""
    reverse_qn(qn::QN) -> QN

Reverse (negate) a quantum number for operator adjoint.

For QN_NSz: negates both components.
For QN_N: negates particle number.

Example:
```julia
qn = QN_NSz(1, 1)
reverse_qn(qn)  # QN_NSz(-1, -1)
```
"""
reverse_qn(qn::QN_NSz) = QN_NSz(-qn.n, -qn.sz)
reverse_qn(qn::QN_N) = QN_N(-qn.n)

"""
    zero_qn(::Type{QN}) -> QN

Return the identity element (zero) for a quantum number type.

Example:
```julia
zero_qn(QN_NSz)  # QN_NSz(0, 0)
zero_qn(QN_N)    # QN_N(0)
```
"""
zero_qn(::Type{QN_NSz}) = QN_NSz(0, 0)
zero_qn(::Type{QN_N}) = QN_N(0)

# ============================================================================
# Ordering (for deterministic sector ordering)
# ============================================================================

"""
    Base.isless(a::QN, b::QN) -> Bool

Define ordering for quantum numbers (for deterministic sector sorting).

For QN_NSz: sort by (n, sz) lexicographically.
For QN_N: sort by n.
"""
function Base.isless(a::QN_NSz, b::QN_NSz)
    if a.n != b.n
        return a.n < b.n
    else
        return a.sz < b.sz
    end
end

Base.isless(a::QN_N, b::QN_N) = a.n < b.n
