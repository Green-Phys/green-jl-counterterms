module QMBToolkitSOEXCA

include("util/InputParams.jl")
include("util/io.jl")
include("util/fill_model_params.jl")

export run

"""
    run(input_file::String)

Main entry point for the solver. Reads parameters from the input file
and starts the calculation.
"""
function run(input_file::String)
    # 1. Read the input parameters
    params = read_input(input_file)
    fill_model_params!(params.model)

    # 2. Start the calculation using the parameters
    # welcome message
    println("========================================")
    println("  QMBToolkitSOEXCA Solver Starting Up  ")
    println("========================================")
    println()
    print_parameters(params)
    
    # In the future, you would call your main solver function here, e.g.:
    # solve(params)
end

end