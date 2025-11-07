using SparseArrays

"""
    DLRParams

Parameters for the DLR expansion.
"""
struct DLRParams
    lambda::Float64
    epsilon::Float64
end

"""
    PropagatorParams

Parameters for the propagator.
"""
struct PropagatorParams
    expansion_order::Int
end

"""
    GreenParams

Parameters for the Green's function.
"""
struct GreenParams
    expansion_order::Int
end

"""
    SolverParams

Parameters for the solver.
"""
struct SolverParams
    propagator::PropagatorParams
    green::GreenParams
    dlr::DLRParams
end

"""
    ModelParams

Parameters for the model.
"""
mutable struct ModelParams
    beta::Float64
    hopping_file::String
    interaction_file::String
    hybridization_file::String
    n_flavors::Int
    n_fock::Int # 2^n_flavors
    ann_opts::Vector{SparseMatrixCSC{Int8,Int}}
    cre_opts::Vector{SparseMatrixCSC{Int8,Int}}
    evals::Vector{Float64}
    evecs::Matrix{ComplexF64}
    Delta::Array{Float64,3}
end

# outer constructor which only requires the first four fields
ModelParams(beta::Float64, hopping_file::String, interaction_file::String, hybridization_file::String) =
    ModelParams(beta, hopping_file, interaction_file, hybridization_file, 0, 0, SparseMatrixCSC{Int8,Int}[], SparseMatrixCSC{Int8,Int}[], Float64[], ComplexF64[;;], Array{Float64,3}(undef, 0,0,0))

"""
    InputParams

All input parameters.
"""
struct InputParams
    model::ModelParams
    solver::SolverParams
end
