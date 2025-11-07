using SparseArrays
using Lehmann

"""
    ModelData

Holds the derived data for the model, such as the Hamiltonian, operators, and hybridization function.
"""
struct ModelData
    n_flavors::Int
    n_fock::Int # 2^n_flavors
    ann_opts::Vector{SparseMatrixCSC{Int8,Int}}
    cre_opts::Vector{SparseMatrixCSC{Int8,Int}}
    evals::Vector{Float64}
    evecs::Matrix{ComplexF64}
    Delta::Array{Float64,3}
end

"""
    DLRData

Holds the DLR grid and related data.
"""
struct DLRData
    grid::DLRGrid
end

"""
    CalculationData

Holds the data for the calculation, such as the Green's function, propagator, etc.
"""
struct CalculationData
    green::Array{Float64, 3}
    propagator::Array{Float64, 3}
    bare_propagator::Array{Float64, 3}
    self_energy::Array{Float64, 3}
end

"""
    SolverData

Holds all the data required for the solver calculation.
"""
struct SolverData
    model::ModelData
    dlr::DLRData
    calc::CalculationData
end
