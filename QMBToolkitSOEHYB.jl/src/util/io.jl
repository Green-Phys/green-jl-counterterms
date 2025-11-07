using TOML

"""
    read_input(file_path::String)

Read input parameters from a TOML file.
"""
function read_input(file_path::String)
    dict = TOML.parsefile(file_path)

    # Check for the existence of top-level keys
    for key in ["model", "solver"]
        if !haskey(dict, key)
            println(stderr, "Error: Missing top-level key '", key, "' in input file.")
            exit(1)
        end
    end

    # Check for keys within the [model] section
    model_keys = ["beta", "hopping_file", "interaction_file", "hybridization_file"]
    for key in model_keys
        if !haskey(dict["model"], key)
            println(stderr, "Error: Missing key '", key, "' in [model] section.")
            exit(1)
        end
    end

    # Check for keys within the [solver] section
    solver_sub_keys = Dict(
        "propagator" => ["expansion_order"],
        "green" => ["expansion_order"],
        "dlr" => ["lambda", "epsilon"]
    )
    for (sub_key, keys) in solver_sub_keys
        if !haskey(dict["solver"], sub_key)
            println(stderr, "Error: Missing section '", sub_key, "' in [solver].")
            exit(1)
        end
        for key in keys
            if !haskey(dict["solver"][sub_key], key)
                println(stderr, "Error: Missing key '", key, "' in [solver.", sub_key, "].")
                exit(1)
            end
        end
    end

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
function print_parameters(params::InputParams)
    println("="^40)
    println(" Model Parameters")
    println("-"^40)
    println("  beta: ", params.model.beta)
    println("  hopping_file: ", params.model.hopping_file)
    println("  interaction_file: ", params.model.interaction_file)
    println("  hybridization_file: ", params.model.hybridization_file)
    println("  n_flavors: ", params.model.n_flavors)
    println("  n_fock: ", params.model.n_fock)
    if length(params.model.evals) >= 4
        println("  evals[1:4]: ", params.model.evals[1:4])
    end
    if size(params.model.Delta, 1) >= 10
        println("  Delta[:10, 1, 1]: ", params.model.Delta[1:10, 1, 1])
        println("  Delta[:10, 1, 2]: ", params.model.Delta[1:10, 1, 2])
    end
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
