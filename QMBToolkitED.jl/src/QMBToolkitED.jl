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

# Types
export Basis, Sector
export Operator, DiagonalOperator, EigenResult

# Core functions
export annihilator, annihilators
export total_qn, fermion_sign

# High-level API
export build_hamiltonian, solve_hamiltonian
export calculate_green_tau, calculate_green_matsubara
export to_matrix, to_vector, get_Z

end # module QMBToolkitED
