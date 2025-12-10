module QMBToolkitED

using LinearAlgebra, SparseArrays

# Include dense submodule
include("Dense/Dense.jl")
using .Dense

# Include block-dense submodule
include("BlockDense/BlockDense.jl")
using .BlockDense

# Include unified interface (provides backwards-compatible API with method parameter)
include("UnifiedInterface.jl")

# Exports - main functions (now from unified interface)
export get_annihilators, build_hamiltonian, solve_hamiltonian
export calculate_green_tau, calculate_green_matsubara, get_Z

# Export conversion functions (from BlockDense module)
export to_dense, from_dense, to_sparse, from_sparse

# Export submodules for direct access
export Dense, BlockDense

"""
    get_Z(evals::Vector{<:Real}, beta::Real; shift_energy::Bool=false)

Calculate the partition function Z = sum(exp(-beta * evals)).
If `shift_energy` is true, the ground state energy is shifted to zero before calculation.
"""
function get_Z(evals::Vector{<:Real}, beta::Real; shift_energy::Bool=false)
    local_evals = copy(evals)
    if shift_energy
        local_evals .-= minimum(local_evals)
    end
    return sum(exp.(-beta .* local_evals))
end

"""
    calculate_green_tau(evals, evecs, annh_ops, beta, n_tau, i, j)

Compute an element of the imaginary-time Green's function matrix G(τ).
Specifically, it calculates -<c_i(τ) c_j^†(0)>.

Works with both dense (Matrix{ComplexF64}) and block-dense (BlockDenseOperator) operators.
"""
function calculate_green_tau(evals::Vector{Float64}, evecs::Matrix{ComplexF64}, annh_ops::Vector{Matrix{ComplexF64}}, beta::Real, n_tau::Int, i::Int, j::Int)
    # Get annihilation operators for indices i and j
    c_i = annh_ops[i]
    c_j = annh_ops[j]
    c_j_dag = c_j'

    # Transform operators to the eigenbasis of the Hamiltonian
    c_i_tilde = evecs' * c_i * evecs
    c_j_dag_tilde = evecs' * c_j_dag * evecs

    # Shift energies so that ground state is at 0
    evals .-= minimum(evals)

    # Partition function
    Z = get_Z(evals, beta, shift_energy=true)

    tau_grid = range(0, beta, length=n_tau)
    gtau = zeros(ComplexF64, n_tau)

    evolve(tau, e) = Diagonal(exp.(-tau .* e))

    for (k, t) in enumerate(tau_grid)
        D1 = evolve(beta - t, evals)
        D2 = evolve(t, evals)
        # Calculates Tr(exp(-(β-τ)H) c_i exp(-τH) c_j^†)
        gtau[k] = tr(D1 * c_i_tilde * D2 * c_j_dag_tilde)
    end

    gtau ./= -Z

    return tau_grid, gtau
end

# Overload for BlockDenseOperator
function calculate_green_tau(evals::Vector{Float64}, evecs::Matrix{ComplexF64}, annh_ops::Vector{BlockDense.BlockDenseOperator}, beta::Real, n_tau::Int, i::Int, j::Int)
    # Convert BlockDenseOperator to dense matrices
    annh_ops_dense = [to_dense(op) for op in annh_ops]
    return calculate_green_tau(evals, evecs, annh_ops_dense, beta, n_tau, i, j)
end

"""
    calculate_green_matsubara(evals, evecs, annh_ops, beta, n_matsubara, i, j)

Compute an element of the Matsubara Green's function G(iω_n) using the Lehmann representation.

Works with both dense (Matrix{ComplexF64}) and block-dense (BlockDenseOperator) operators.
"""
function calculate_green_matsubara(evals::Vector{<:Real}, evecs::Matrix, annh_ops::Vector{Matrix{ComplexF64}}, beta::Real, n_matsubara::Int, i::Int, j::Int)
    nfock = size(evecs, 1)

    # Get annihilation operators for indices i and j
    c_i = annh_ops[i]
    c_j = annh_ops[j]
    c_j_dag = c_j'

    # Transform operators to the eigenbasis of the Hamiltonian
    c_i_tilde = evecs' * c_i * evecs
    c_j_dag_tilde = evecs' * c_j_dag * evecs

    # Shift energies so that ground state is at 0
    evals .-= minimum(evals)

    # Partition function
    Z = get_Z(evals, beta, shift_energy=true)

    # Matsubara frequencies
    omega_n = (2 .* (0:(n_matsubara-1)) .+ 1) .* π ./ beta
    iomega_n = 1im .* omega_n

    g_iwn = zeros(ComplexF64, n_matsubara)

    for m in 1:nfock, n in 1:nfock
        energy_diff = evals[m] - evals[n]
        
        # <m|c_i|n> <n|c_j^†|m>
        matrix_element = c_i_tilde[m, n] * c_j_dag_tilde[n, m]
        
        if abs(matrix_element) > 1e-9
            boltzmann_factor = exp(-beta * evals[m]) + exp(-beta * evals[n])
            
            g_iwn .+= boltzmann_factor * matrix_element ./ (iomega_n .+ energy_diff)
        end
    end

    g_iwn ./= Z

    return iomega_n, g_iwn
end

# Overload for BlockDenseOperator
function calculate_green_matsubara(evals::Vector{<:Real}, evecs::Matrix, annh_ops::Vector{BlockDense.BlockDenseOperator}, beta::Real, n_matsubara::Int, i::Int, j::Int)
    # Convert BlockDenseOperator to dense matrices
    annh_ops_dense = [to_dense(op) for op in annh_ops]
    return calculate_green_matsubara(evals, evecs, annh_ops_dense, beta, n_matsubara, i, j)
end


end # module QMBToolkitED