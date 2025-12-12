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
    is_bold::Bool
end

"""
    GreenParams

Parameters for the Green's function.
"""
struct GreenParams
    expansion_order::Int
    is_bold::Bool
    save_linear_size::Int
end

"""
    SolverParams

Parameters for the solver.
"""
struct SolverParams
    propagator::PropagatorParams
    green::GreenParams
    dlr::DLRParams
    output_path::String
    format::String  # "dense" or "blockdense"
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
    hybridization_poles_file::String
    hybridization_weights_file::String
end

"""
    Params

All input parameters.
"""
struct Params
    model::ModelParams
    solver::SolverParams
end
