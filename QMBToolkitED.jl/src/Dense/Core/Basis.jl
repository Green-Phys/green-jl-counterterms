"""
Dense basis for quantum many-body systems.

The dense basis represents the full Hilbert space without symmetry blocking.
This provides API symmetry with BlockDenseBasis while being a simple wrapper.
"""

"""
    DenseBasis

A many-body basis representing the full Hilbert space.

Fields:
- `nflavors::Int`: Number of orbitals/flavors
- `dim::Int`: Total Hilbert space dimension (== 2^nflavors)

This is the dense analog of `BlockDenseBasis`, providing a consistent API
across both methods.

Example:
```julia
# Create basis for 4 orbitals
basis = make_basis_dense(4)
println("Dimension: ", basis.dim)  # 16
```
"""
struct DenseBasis
    nflavors::Int
    dim::Int

    function DenseBasis(nflavors::Int)
        @assert nflavors > 0 "nflavors must be positive"
        new(nflavors, 1 << nflavors)
    end
end

"""
    make_basis_dense(nflavors::Int) -> DenseBasis

Construct a dense basis for a Fock space with `nflavors` orbitals.

Arguments:
- `nflavors::Int`: Number of orbitals

Returns:
- `DenseBasis`: The full Hilbert space basis

Example:
```julia
basis = make_basis_dense(3)
# basis.dim == 8
```
"""
make_basis_dense(nflavors::Int) = DenseBasis(nflavors)

# Display
function Base.show(io::IO, basis::DenseBasis)
    print(io, "DenseBasis(nflavors=$(basis.nflavors), dim=$(basis.dim))")
end
