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
- `Base.:(+)` - addition of quantum numbers
- `Base.:(-)` - negation (for operator adjoint)
- `Base.zero` - identity element
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
qn_total = qn_up + qn_down  # QN_NSz(2, 0)
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
qn_total = qn1 + qn2  # QN_N(2)
```
"""
struct QN_N <: QuantumNumber
    n::Int   # Particle number
end

"""
    QN_Trivial

Trivial quantum number for systems with no conserved quantities.

All states have the same quantum number, resulting in a single sector
spanning the entire Hilbert space. This is equivalent to the dense
representation but uses the block-dense infrastructure.

Example:
```julia
# Create a 4-flavor system with no symmetry
flavor_qns = [QN_Trivial() for _ in 1:4]
basis = Basis(flavor_qns)
# Results in 1 sector with dimension 2^4 = 16
```
"""
struct QN_Trivial <: QuantumNumber end

# ============================================================================
# Interface Implementation: Equality and Hashing
# ============================================================================

Base.:(==)(a::QN_NSz, b::QN_NSz) = (a.n == b.n) && (a.sz == b.sz)
Base.:(==)(a::QN_N, b::QN_N) = (a.n == b.n)
Base.:(==)(::QN_Trivial, ::QN_Trivial) = true

Base.hash(qn::QN_NSz, h::UInt) = hash((qn.n, qn.sz), h)
Base.hash(qn::QN_N, h::UInt) = hash(qn.n, h)
Base.hash(::QN_Trivial, h::UInt) = hash(:QN_Trivial, h)

# ============================================================================
# Interface Implementation: Display
# ============================================================================

function Base.show(io::IO, qn::QN_NSz)
    print(io, "QN_NSz(n=", qn.n, ", sz=", qn.sz, ")")
end

function Base.show(io::IO, qn::QN_N)
    print(io, "QN_N(n=", qn.n, ")")
end

function Base.show(io::IO, ::QN_Trivial)
    print(io, "QN_Trivial()")
end

# ============================================================================
# Quantum Number Arithmetic
# ============================================================================

"""
    Base.:(+)(qn1::QN, qn2::QN) -> QN

Add two quantum numbers.

For QN_NSz: adds both particle number and spin components.
For QN_N: adds particle numbers.

Example:
```julia
qn1 = QN_NSz(1, 1)
qn2 = QN_NSz(1, -1)
qn1 + qn2  # QN_NSz(2, 0)
```
"""
Base.:(+)(qn1::QN_NSz, qn2::QN_NSz) = QN_NSz(qn1.n + qn2.n, qn1.sz + qn2.sz)
Base.:(+)(qn1::QN_N, qn2::QN_N) = QN_N(qn1.n + qn2.n)
Base.:(+)(::QN_Trivial, ::QN_Trivial) = QN_Trivial()

"""
    Base.:(-)(qn::QN) -> QN

Negate a quantum number (for operator adjoint).

For QN_NSz: negates both components.
For QN_N: negates particle number.

Example:
```julia
qn = QN_NSz(1, 1)
-qn  # QN_NSz(-1, -1)
```
"""
Base.:(-)(qn::QN_NSz) = QN_NSz(-qn.n, -qn.sz)
Base.:(-)(qn::QN_N) = QN_N(-qn.n)
Base.:(-)(::QN_Trivial) = QN_Trivial()

"""
    Base.zero(::Type{QN}) -> QN

Return the identity element (zero) for a quantum number type.

Example:
```julia
zero(QN_NSz)  # QN_NSz(0, 0)
zero(QN_N)    # QN_N(0)
```
"""
Base.zero(::Type{QN_NSz}) = QN_NSz(0, 0)
Base.zero(::Type{QN_N}) = QN_N(0)
Base.zero(::Type{QN_Trivial}) = QN_Trivial()

# ============================================================================
# Ordering (for deterministic sector ordering)
# ============================================================================

"""
    Base.isless(a::QN, b::QN) -> Bool

Define ordering for quantum numbers (for deterministic sector sorting).

For QN_NSz: sort by (n, sz) lexicographically.
For QN_N: sort by n.
"""
Base.isless(a::QN_NSz, b::QN_NSz) = (a.n, a.sz) < (b.n, b.sz)

Base.isless(a::QN_N, b::QN_N) = a.n < b.n

# QN_Trivial: only one value, so never less than itself
Base.isless(::QN_Trivial, ::QN_Trivial) = false
