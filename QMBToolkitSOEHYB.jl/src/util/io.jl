using TOML
using Lehmann

"""
    read_input(file_path::String)

Read input parameters from a TOML file.
"""
function read_input(file_path::String)
    dict = TOML.parsefile(file_path)

    @assert dict["model"]["beta"] > 0 "beta must be positive"
    @assert dict["solver"]["propagator"]["expansion_order"] >= 0 "expansion_order must be non-negative"
    @assert dict["solver"]["green"]["expansion_order"] >= 0 "expansion_order must be non-negative"
    @assert dict["solver"]["dlr"]["rtol"] > 0 "rtol must be positive"
    @assert dict["solver"]["dlr"]["Euv"] > 0 "Euv must be positive"

    model_params = ModelParams(
        dict["model"]["beta"],
        dict["model"]["hopping_file"],
        dict["model"]["interaction_file"],
        dict["model"]["hybridization_file"]
    )

    propagator_params = PropagatorParams(
        dict["solver"]["propagator"]["expansion_order"]
    )

    green_params = GreenParams(
        dict["solver"]["green"]["expansion_order"]
    )

    dlr_params = DLRParams(
        dict["solver"]["dlr"]["Euv"],
        dict["solver"]["dlr"]["rtol"]
    )

    solver_params = SolverParams(
        propagator_params,
        green_params,
        dlr_params
    )

    return Params(model_params, solver_params)
end

"""
    print_parameters(params::Params)

Print parameters in a formatted way.
"""
function print_parameters(params::Params)
    println("="^40)
    println(" Input Parameters")
    println("-"^40)
    println("  [model]")
    println("    beta: ", params.model.beta)
    println("    hopping_file: ", params.model.hopping_file)
    println("    interaction_file: ", params.model.interaction_file)
    println("    hybridization_file: ", params.model.hybridization_file)
    println("  [solver]")
    println("    [propagator]")
    println("      expansion_order: ", params.solver.propagator.expansion_order)
    println("    [green]")
    println("      expansion_order: ", params.solver.green.expansion_order)
    println("    [dlr]")
    println("      Euv: ", params.solver.dlr.Euv)
    println("      rtol: ", params.solver.dlr.rtol)
    println("="^40)
end

"""
    print_init_data(solver_data::SolverData)

Print the contents of a `SolverData` object.
"""
function print_init_data(solver_data::SolverData)
    println("="^40)
    println(" Model Data")
    println("-"^40)
    println("  n_flavors: ", solver_data.model.n_flavors)
    println("  n_fock: ", solver_data.model.n_fock)
    if length(solver_data.model.evals) >= 4
        println("  evals[1:4]: ", solver_data.model.evals[1:4])
    end
    if size(solver_data.model.Delta, 1) >= 10
        println("  Delta[:10, 1, 1]: ", solver_data.model.Delta[1:10, 1, 1])
        println("  Delta[:10, 1, 2]: ", solver_data.model.Delta[1:10, 1, 2])
    end
    println("-"^40)
    println(" DLR Data")
    println("-"^40)
    println("  # of DLR coefficients: ", solver_data.model.dlr_grid.size)
    println("="^40)
    println(" DLR Grid Points (first 5):")
    for i in 1:min(5, solver_data.model.dlr_grid.size)
        println("  w[", i, "] = ", solver_data.model.dlr_grid.omega[i])
    end
    println("Maximum reconstruction error in Delta: ", solver_data.calc.Delta_error)
    println("="^40)
end
