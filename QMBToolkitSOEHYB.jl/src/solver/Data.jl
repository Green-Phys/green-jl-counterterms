using SparseArrays
using Lehmann

"""
    ModelData

Holds the derived data for the model, such as the Hamiltonian, operators, and hybridization function.
"""
struct ModelData
    beta::Float64
    n_flavors::Int
    n_fock::Int # 2^n_flavors
    ann_opts::Vector{SparseMatrixCSC{Int8,Int}}
    cre_opts::Vector{SparseMatrixCSC{Int8,Int}}
    evals::Vector{Float64}
    evecs::Matrix{ComplexF64}
    Delta::Array{Float64,3}
    tau_grid_Delta::Array{Float64,1}
    dlr_grid::DLRGrid
end

"""
    CalculationData

Holds the data for the calculation, such as the Green's function, propagator, etc.
"""
mutable struct CalculationData
    Delta_error::Float64
    Delta_dlr::Array{Float64,3}
    green_dlr::Array{Float64, 3}
    propagator_dlr::Array{Float64, 3}
    bare_propagator_dlr::Array{Float64, 3}
    self_energy_dlr::Array{Float64, 3}
end

"""
    SolverData

Holds all the data required for the solver calculation.
"""
struct SolverData
    model::ModelData
    calc::CalculationData
end
