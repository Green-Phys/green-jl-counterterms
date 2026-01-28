module QMBToolkitED

using LinearAlgebra

# Core types
include("QuantumNumbers.jl")
include("Basis.jl")
include("Operators.jl")

# Operations
include("Algebra.jl")
include("Conversion.jl")

# High-level API
include("Hamiltonian.jl")
include("Diagonalization.jl")
include("Greens.jl")

# Partition function utility
function get_Z(evals::Vector{<:Real}, beta::Real; shift_energy::Bool=false)
    local_evals = copy(evals)
    if shift_energy
        local_evals .-= minimum(local_evals)
    end
    return sum(exp.(-beta .* local_evals))
end

# Quantum numbers
export QuantumNumber, QN_NSz, QN_N, QN_Trivial
export combine_qn, reverse_qn, zero_qn

# Types
export Basis, Sector, OperatorFlux
export Operator, DiagonalOperator, EigenResult

# Core functions
export make_basis, get_annihilator, get_annihilators
export sum_occupied_sites, compute_fermion_sign

# High-level API
export build_hamiltonian, solve_hamiltonian
export calculate_green_tau, calculate_green_matsubara
export to_matrix, to_vector, get_Z

end # module QMBToolkitED
