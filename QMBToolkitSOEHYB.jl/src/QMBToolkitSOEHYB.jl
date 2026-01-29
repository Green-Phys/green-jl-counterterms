module QMBToolkitSOEHYB

using MPI

include("util/Params.jl")

# Format interface - defines the abstract interface for format-agnostic operations
include("format/interface.jl")

# Propagator types - must be loaded before Data.jl and other files that use them
include("format/GreenFunction.jl")
include("format/propagator.jl")

include("solver/Data.jl")
include("util/io.jl")
include("util/init_data.jl")
include("solver/solve.jl")
include("diagram/DiagramStructure.jl")
include("diagram/print_diagram.jl")
include("diagram/gen_common.jl")
include("diagram/DiagramConfig.jl")

# Format operations
include("format/dlr.jl")
include("format/operator_ops.jl")
include("format/hybridization.jl")
include("format/backbone_buffers.jl")

# Unified backbone evaluation (uses interface functions for format dispatch)
include("diagram/backbone_eval.jl")

include("diagram/eval_common.jl")
include("diagram/eval_propagator.jl")
include("diagram/eval_green.jl")
include("diagram/eval_selfenergy.jl")

export run

"""
    run(input_file::String)

Main entry point for the solver. Reads parameters from the input file
and starts the calculation.
"""
function run(input_file::String)
    mpi_was_initialized = MPI.Initialized()
    if !mpi_was_initialized
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if rank == 0
        print_startup_message()
    end

    params = read_input(input_file)

    model_data = init_model_data(params.model, params.solver.dlr)

    solver_data = init_solver_data(model_data) # solver_data contains model_data and calculation_data

    if rank == 0
        print_parameters(params)
        print_init_data(solver_data)
    end

    solve(solver_data, params.solver, comm)

    if !mpi_was_initialized
        MPI.Finalize()
    end
end

end
