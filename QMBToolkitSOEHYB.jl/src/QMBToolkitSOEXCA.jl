module QMBToolkitSOEXCA

include("util/Params.jl")
include("solver/Data.jl")
include("solver/solver.jl")
include("util/io.jl")
include("util/init_data.jl")
include("util/others.jl")
include("diagram/DiagramStructure.jl")
include("diagram/gen_green.jl")
include("diagram/print_diagram.jl")

export run, print_diagram

"""
    run(input_file::String)

Main entry point for the solver. Reads parameters from the input file
and starts the calculation.
"""
function run(input_file::String)
    # 1. Read the input parameters
    params = read_input(input_file)

    # 2. Create model data from files
    model_data = create_model_data(params.model, params.solver.dlr)

    # 3. Create solver data
    solver_data = create_solver_data(model_data)

    # 4. Start the calculation using the parameters
    # welcome message
    println("========================================")
    println("  QMBToolkitSOEXCA Solver Starting Up  ")
    println("========================================")
    println()
    print_parameters(params)
    print_init_data(solver_data)
    
    # In the future, you would call your main solver function here, e.g.:
    solve(solver_data, params.solver)
end

end