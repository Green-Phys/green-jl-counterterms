using TOML

"""
    read_input(file_path::String)

Read input parameters from a TOML file.
"""
function _read_input(file_path::String)
    dict = TOML.parsefile(file_path)

    model_params = ModelParams(
        dict["model"]["beta"],
        dict["model"]["hopping_file"],
        dict["model"]["interaction_file"],
        dict["model"]["hybridization_file"]
    )

    _fill_model_params!(model_params)


    propagator_params = PropagatorParams(
        dict["solver"]["propagator"]["expansion_order"]
    )

    green_params = GreenParams(
        dict["solver"]["green"]["expansion_order"]
    )

    dlr_params = DLRParams(
        dict["solver"]["dlr"]["lambda"],
        dict["solver"]["dlr"]["epsilon"]
    )

    solver_params = SolverParams(
        propagator_params,
        green_params,
        dlr_params
    )

    return InputParams(model_params, solver_params)
end

"""
    print_parameters(params::InputParams)

Print parameters in a formatted way.
"""
function _print_parameters(params::InputParams)
    println("="^40)
    println(" Model Parameters")
    println("-"^40)
    println("  beta: ", params.model.beta)
    println("  hopping_file: ", params.model.hopping_file)
    println("  interaction_file: ", params.model.interaction_file)
    println("  hybridization_file: ", params.model.hybridization_file)
    println("  n_flavors: ", params.model.n_flavors)
    println("  n_fock: ", params.model.n_fock)
    println("  evals[1:4]: ", params.model.evals[1:4])
    println("  Delta[:10, 1, 1]: ", params.model.Delta[1:10, 1, 1])
    println("  Delta[:10, 1, 2]: ", params.model.Delta[1:10, 1, 2])
    println(" Solver Parameters")
    println("-"^40)
    println("  [propagator]")
    println("    expansion_order: ", params.solver.propagator.expansion_order)
    println("  [green]")
    println("    expansion_order: ", params.solver.green.expansion_order)
    println("  [dlr]")
    println("    lambda: ", params.solver.dlr.lambda)
    println("    epsilon: ", params.solver.dlr.epsilon)
    println("="^40)
end
