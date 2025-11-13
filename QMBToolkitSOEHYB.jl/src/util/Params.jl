using SparseArrays

"""
    DLRParams

Parameters for the DLR expansion.
"""
struct DLRParams
    Euv::Float64
    rtol::Float64
end

"""
    ExpansionParams

Parameters for the propagator.
"""
struct ExpansionParams
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
    expansion::ExpansionParams
    green::GreenParams
    dlr::DLRParams
    output_path::String
end

"""
    ModelParams

Parameters for the model.
"""
struct ModelParams
    beta::Float64
    file_path::String
    hopping_file::String
    interaction_file::String
    hybridization_file::String
    poles_energies_file::String
    poles_hybs_file::String
end

"""
    Params

All input parameters.
"""
struct Params
    model::ModelParams
    solver::SolverParams
end
