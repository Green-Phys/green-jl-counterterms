module QMBToolkitSOEXCA

include("util/Params.jl")
include("solver/Data.jl")
include("util/io.jl")
include("util/init_data.jl")
include("util/others.jl")
include("solver/solve.jl")
include("diagram/DiagramStructure.jl")
include("diagram/gen_common.jl")
include("diagram/eval_common.jl")
include("diagram/gen_eval_propagator.jl")
include("diagram/gen_eval_green.jl")

export run

"""
    run(input_file::String)

Main entry point for the solver. Reads parameters from the input file
and starts the calculation.
"""
function run(input_file::String)
    # 1. Read the input parameters
    params = read_input(input_file)

    # 2. Initialize model data from files
    model_data = init_model_data(params.model, params.solver.dlr)

    # 3. Initialize solver data
    solver_data = init_solver_data(model_data)

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