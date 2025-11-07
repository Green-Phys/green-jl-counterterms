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
struct ModelParams
    beta::Float64
    hopping_file::String
    interaction_file::String
    hybridization_file::String
end

"""
    Params

All input parameters.
"""
struct Params
    model::ModelParams
    solver::SolverParams
end
