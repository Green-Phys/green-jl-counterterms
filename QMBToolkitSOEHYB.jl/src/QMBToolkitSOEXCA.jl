module QMBToolkitSOEXCA

include("util/Params.jl")
include("solver/Data.jl")
include("util/io.jl")
include("util/init_data.jl")

export run

"""
    run(input_file::String)

Main entry point for the solver. Reads parameters from the input file
and starts the calculation.
"""
function run(input_file::String)
    # 1. Read the input parameters
    params = read_input(input_file)

    # 2. Create model data from files
    model_data = create_model_data(params.model)

    # 3. Create DLR data
    dlr_data = create_dlr_data(params.solver.dlr, params.model)

    # 4. Initialize calculation data
    n_tau, n_orb, _ = size(model_data.Delta)
    calc_data = CalculationData(
        zeros(n_tau, n_orb, n_orb),
        zeros(n_tau, n_orb, n_orb),
        zeros(n_tau, n_orb, n_orb),
        zeros(n_tau, n_orb, n_orb)
    )
    solver_data = SolverData(model_data, dlr_data, calc_data)

    # 5. Start the calculation using the parameters
    # welcome message
    println("========================================")
    println("  QMBToolkitSOEXCA Solver Starting Up  ")
    println("========================================")
    println()
    print_parameters(params)
    print_model_data(solver_data.model)
    print_dlr_data(solver_data.dlr)
    
    # In the future, you would call your main solver function here, e.g.:
    # solve(solver_data, params.solver)
end

end