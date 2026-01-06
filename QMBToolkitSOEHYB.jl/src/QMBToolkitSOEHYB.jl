module QMBToolkitSOEHYB

using MPI

include("util/Params.jl")
include("solver/HilbertSpaceFormat.jl")

# Format interface - defines the abstract interface for format-agnostic operations
include("format/interface.jl")

# Propagator types - must be loaded before Data.jl and other files that use them
include("format/dense/DensePropagator.jl")
include("format/blockdense/BlockDensePropagator.jl")

include("solver/Data.jl")
include("util/io.jl")
include("util/init_data.jl")
include("solver/solve.jl")
include("diagram/DiagramStructure.jl")
include("diagram/print_diagram.jl")
include("diagram/gen_common.jl")

# Dense format operations
include("format/dense/dlr.jl")
include("format/dense/operator_ops.jl")
include("format/dense/hybridization.jl")
include("format/dense/propagator.jl")

# BlockDense format operations
include("format/blockdense/dlr.jl")
include("format/blockdense/operator_ops.jl")
include("format/blockdense/hybridization.jl")
include("format/blockdense/propagator.jl")

# Unified backbone evaluation (uses interface functions for format dispatch)
include("diagram/backbone_eval.jl")

include("diagram/eval_common.jl")
include("diagram/gen_propagator.jl")
include("diagram/eval_propagator.jl")
include("diagram/gen_green.jl")
include("diagram/eval_green.jl")
include("diagram/gen_selfenergy.jl")
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
        println("========================================")
        println("  QMBToolkitSOEHYB Solver Starting Up  ")
        println("========================================")
    end

    params = read_input(input_file)

    if params.solver.Hilbert_space_format == "dense"
        model_data = init_model_data(params.model, params.solver.dlr, DenseFormat)
    elseif params.solver.Hilbert_space_format == "blockdense"
        model_data = init_model_data(params.model, params.solver.dlr, BlockDenseFormat)
    else
        error("Unknown Hilbert_space_format: $(params.solver.Hilbert_space_format)")
    end

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
