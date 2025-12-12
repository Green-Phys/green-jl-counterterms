module QMBToolkitSOEXCA

using MPI

include("util/Params.jl")
include("solver/Representation.jl")
include("solver/Data.jl")
include("util/io.jl")
include("util/init_data.jl")
include("util/others.jl")
include("solver/solve.jl")
include("diagram/DiagramStructure.jl")
include("diagram/print_diagram.jl")
include("diagram/gen_common.jl")

# Dense operations (current implementation)
include("dense/operator_ops.jl")
include("dense/hybridization.jl")
include("dense/convolution_eval.jl")
include("dense/propagator.jl")

# BlockDense operations (Phase 5 implementation)
include("blockdense/algebra.jl")
include("blockdense/operator_ops.jl")
include("blockdense/hybridization.jl")
include("blockdense/propagator.jl")
include("blockdense/convolution_eval.jl")

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
    # Initialize MPI (only if not already initialized)
    mpi_was_initialized = MPI.Initialized()
    if !mpi_was_initialized
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Only rank 0 prints output
    if rank == 0
        println("========================================")
        println("  QMBToolkitSOEXCA Solver Starting Up  ")
        println("========================================")
    end

    # 1. Read the input parameters
    params = read_input(input_file)

    # 2. Initialize model data from files (format-dependent)
    if params.solver.format == "dense"
        model_data = init_model_data(params.model, params.solver.dlr)
    elseif params.solver.format == "blockdense"
        model_data = init_model_data_blockdense(params.model, params.solver.dlr)
    else
        error("Unknown format: $(params.solver.format)")
    end

    # 3. Initialize solver data
    solver_data = init_solver_data(model_data)

    if rank == 0
        print_parameters(params)
        print_init_data(solver_data)
    end

    # 4. Start the calculation using the parameters
    solve(solver_data, params.solver, comm)

    # Finalize MPI (only if we initialized it in this function)
    if !mpi_was_initialized
        MPI.Finalize()
    end
end

end