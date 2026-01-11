using TOML
using Printf
using HDF5

"""
    resolve_path(path::String, input_file_dir::String)

Resolve a path relative to the input file directory.
If the path is absolute, return it as-is.
If the path is relative, join it with the input file directory.
"""
function resolve_path(path::String, input_file_dir::String)
    if isabspath(path)
        return path
    else
        return joinpath(input_file_dir, path)
    end
end

"""
    read_input(file_path::String)

Read input parameters from a TOML file.
"""
function read_input(file_path::String)
    dict = TOML.parsefile(file_path)

    # Validate input parameters
    dict["model"]["beta"] > 0 || error("beta must be positive, got $(dict["model"]["beta"])")
    dict["solver"]["propagator"]["expansion_order"] >= 0 ||
        error("propagator expansion_order must be non-negative, got $(dict["solver"]["propagator"]["expansion_order"])")
    dict["solver"]["green"]["expansion_order"] >= 1 ||
        error("green expansion_order must be at least 1, got $(dict["solver"]["green"]["expansion_order"])")
    dict["solver"]["dlr"]["rtol"] > 0 ||
        error("dlr rtol must be positive, got $(dict["solver"]["dlr"]["rtol"])")
    dict["solver"]["dlr"]["Euv"] > 0 ||
        error("dlr Euv must be positive, got $(dict["solver"]["dlr"]["Euv"])")

    # Get the directory containing the input file for resolving relative paths
    input_dir = dirname(abspath(file_path))

    # Resolve paths relative to input file directory
    model_file_path = resolve_path(dict["model"]["file_path"], input_dir)
    output_path = resolve_path(dict["solver"]["output_path"], input_dir)

    model_params = ModelParams(
        dict["model"]["beta"],
        model_file_path,
        dict["model"]["hopping_file"],
        dict["model"]["interaction_file"],
        dict["model"]["hybridization_poles_file"],
        dict["model"]["hybridization_weights_file"]
    )

    propagator_params = PropagatorParams(
        dict["solver"]["propagator"]["expansion_order"],
        dict["solver"]["propagator"]["is_bold"]
    )

    green_params = GreenParams(
        dict["solver"]["green"]["expansion_order"],
        dict["solver"]["green"]["is_bold"],
        dict["solver"]["green"]["save_linear_size"]
    )

    dlr_params = DLRParams(
        dict["solver"]["dlr"]["Euv"],
        dict["solver"]["dlr"]["rtol"]
    )

    # Read Hilbert_space_format (default to "dense" for backward compatibility)
    Hilbert_space_format = get(dict["solver"], "Hilbert_space_format", "dense")
    @assert Hilbert_space_format in ["dense", "blockdense"] "Hilbert_space_format must be 'dense' or 'blockdense', got: $Hilbert_space_format"

    solver_params = SolverParams(
        propagator_params,
        green_params,
        dlr_params,
        output_path,
        Hilbert_space_format
    )

    return Params(model_params, solver_params)
end

"""
    print_parameters(params::Params)

Print parameters in a formatted way.
"""
function print_parameters(params::Params)
    println()
    println("--- Input Parameters ---")
    println(" [model]")
    println("   beta: ", params.model.beta)
    println("   file_path: ", params.model.file_path)
    println("   hopping_file: ", params.model.hopping_file)
    println("   interaction_file: ", params.model.interaction_file)
    println("   hybridization_poles_file: ", params.model.hybridization_poles_file)
    println("   hybridization_weights_file: ", params.model.hybridization_weights_file)
    println(" [solver]")
    println("   output_path: ", params.solver.output_path)
    println("   Hilbert_space_format: ", params.solver.Hilbert_space_format)
    println("  [expansion]")
    println("    is_bold: ", params.solver.propagator.is_bold)
    println("    expansion_order: ", params.solver.propagator.expansion_order)
    println("  [green]")
    println("    is_bold: ", params.solver.green.is_bold)
    println("    expansion_order: ", params.solver.green.expansion_order)
    println("    save_linear_size: ", params.solver.green.save_linear_size)
    println("  [dlr]")
    println("    Euv: ", params.solver.dlr.Euv)
    println("    rtol: ", params.solver.dlr.rtol)
end

"""
    print_init_data(solver_data::SolverData)

Print the contents of a `SolverData` object.
"""
function print_init_data(solver_data::SolverData)
    println()
    println("--- Model Data ---")
    println(" n_flavors: ", solver_data.model.n_flavors)
    println(" n_fock: ", solver_data.model.n_fock)
    println(" n_flavors_physical: ", solver_data.model.n_flavors_physical)
    println(" # of SOE for hybridization function: ", length(solver_data.model.hyb.exponents))
    println(" E_shift applied to Hamiltonian: ", solver_data.model.E_shift)
    println()
    println("--- DLR Data ---")
    println(" # of DLR coefficients: ", solver_data.model.dlr_grid.size)
    println(" Maximum reconstruction error in bare propagator: ", solver_data.model.bare_propagator_err)
end

"""
    save_data(model_data, calc_data, file_path, total_wall_time, total_cpu_time)

Save all calculation results to an HDF5 file.

Saves data in organized groups:

  - /green/linear_grid/green_tau: Green's function on uniform linear grid
  - /green/linear_grid/tau: Uniform linear tau grid
  - /green/dlr_grid/green_tau: Green's function on DLR tau grid
  - /green/dlr_grid/tau: DLR tau grid points
  - /green/dlr_grid/coeffs: DLR expansion coefficients
  - /model/E_shift: Energy shift applied to Hamiltonian
  - /model/beta: Inverse temperature
  - /calculation/partition_function: Final partition function
  - /calculation/partition_function_by_order: Partition function after each expansion order
  - /timing/total_wall_time: Total wall clock time in seconds
  - /timing/total_cpu_time: Total CPU time across all MPI ranks in seconds
"""
function save_data(
        model_data::ModelData,
        calc_data::CalculationData,
        file_path::String,
        total_wall_time::Float64,
        total_cpu_time::Float64
)
    h5open(file_path, "w") do file
        # Save Green's function - linear grid
        green_linear_group = create_group(file, "green/linear_grid")
        green_linear_group["green_tau"] = calc_data.green_linear_grid.data
        green_linear_group["tau"] = calc_data.tau_linear_grid

        # Save Green's function - DLR grid
        green_dlr_group = create_group(file, "green/dlr_grid")
        green_dlr_group["green_tau"] = calc_data.green.vals.data
        green_dlr_group["tau"] = model_data.dlr_grid.τ
        green_dlr_group["coeffs"] = calc_data.green.coeffs.data

        # Save Green's function by order (on linear grid)
        # Note: green_by_order is stored on DLR grid, need to interpolate to linear grid
        green_by_order_group = create_group(file, "green/by_order")
        for (i, green_at_order_dlr) in enumerate(calc_data.green_by_order)
            # Convert from DLR grid to DLR coefficients
            green_coeffs = tau2dlr(model_data.dlr_grid, green_at_order_dlr, model_data.dlr_grid.τ)
            # Interpolate to linear grid
            green_at_order_linear = dlr2tau(model_data.dlr_grid, green_coeffs, calc_data.tau_linear_grid)
            green_by_order_group["green_order_$i"] = green_at_order_linear.data
        end

        # Save model parameters
        model_group = create_group(file, "model")
        model_group["E_shift"] = model_data.E_shift
        model_group["beta"] = model_data.beta
        model_group["Z_counterterms"] = model_data.Z_counterterms

        # Save calculation results
        calc_group = create_group(file, "partition_function")
        calc_group["partition_function"] = calc_data.partition_function_by_order[end]
        calc_group["partition_function_by_order"] = calc_data.partition_function_by_order
        partition_function_corrected = exp(-model_data.E_shift*model_data.beta) *
                                       calc_data.partition_function_by_order[end] /
                                       model_data.Z_counterterms
        calc_group["partition_function_corrected"] = partition_function_corrected

        # Save timing information
        timing_group = create_group(file, "timing")
        timing_group["total_wall_time"] = total_wall_time
        timing_group["total_cpu_time"] = total_cpu_time
    end
end

"""
    save_results(solver_data::SolverData, solver_params::SolverParams, total_wall_time::Float64, total_cpu_time::Float64)

Process Green's function data and save to HDF5 file. Returns nothing.
Prints summary information about the results.
"""
function save_results(solver_data::SolverData, solver_params::SolverParams,
        total_wall_time::Float64, total_cpu_time::Float64)
    println()
    println("--- Post-processing and saving data ---")

    # Compute linear grid and store in CalculationData
    # IMPORTANT: This must be done BEFORE calling save_data since save_data uses the linear grid
    solver_data.calc.tau_linear_grid = collect(LinRange(
        0.0, solver_data.model.beta, solver_params.green.save_linear_size))
    green_linear = dlr2tau(
        solver_data.model.dlr_grid, solver_data.calc.green.coeffs,
        solver_data.calc.tau_linear_grid)
    solver_data.calc.green_linear_grid = green_linear

    # Print summary
    println("First 4 linear grid tau points:")
    println(solver_data.calc.tau_linear_grid[1:4])
    println("Green's function G[1:4, 1,1] on linear grid:")
    println(solver_data.calc.green_linear_grid.data[1:4, 1, 1])
    println("Head + Tail (linear grid):")
    println(solver_data.calc.green_linear_grid.data[1, 1, 1] +
            solver_data.calc.green_linear_grid.data[end, 1, 1])

    # Save all data
    output_file = joinpath(solver_params.output_path, "sim.h5")
    save_data(solver_data.model, solver_data.calc, output_file, total_wall_time, total_cpu_time)
    println("Saved results to: ", output_file)
end

"""
    print_shutdown_message(nprocs, t_gen_prop_wall, t_eval_prop_wall, t_eval_prop_total,
                        t_gen_green_wall, t_eval_green_wall, t_eval_green_total)

Print a formatted timing summary table showing wall times, percentages, and CPU times
for diagram generation and evaluation operations.
"""
function print_shutdown_message(
        nprocs::Int,
        t_gen_prop_wall::Float64,
        t_eval_prop_wall::Float64,
        t_eval_prop_total::Float64,
        t_gen_green_wall::Float64,
        t_eval_green_wall::Float64,
        t_eval_green_total::Float64
)
    println()
    println("--- TIMING SUMMARY ---")
    println("Number of MPI processes: $nprocs")
    println()

    # Calculate total time
    total_wall = t_gen_prop_wall + t_eval_prop_wall + t_gen_green_wall + t_eval_green_wall

    # Print table header
    println("Operation                              | Wall Time (s) | % Total | CPU Time (s)")
    println("---------------------------------------+---------------+---------+-------------")

    # Propagator generation
    prop_gen_pct = 100.0 * t_gen_prop_wall / total_wall
    @printf("Propagator diagram generation          | %13.3f | %6.1f%% | %s\n",
        t_gen_prop_wall, prop_gen_pct, "-")

    # Propagator evaluation
    prop_eval_pct = 100.0 * t_eval_prop_wall / total_wall
    @printf("Propagator diagram evaluation          | %13.3f | %6.1f%% | %11.3f\n",
        t_eval_prop_wall, prop_eval_pct, t_eval_prop_total)

    # Green's function generation
    green_gen_pct = 100.0 * t_gen_green_wall / total_wall
    @printf("Green's function diagram generation    | %13.3f | %6.1f%% | %s\n",
        t_gen_green_wall, green_gen_pct, "-")

    # Green's function evaluation
    green_eval_pct = 100.0 * t_eval_green_wall / total_wall
    @printf("Green's function diagram evaluation    | %13.3f | %6.1f%% | %11.3f\n",
        t_eval_green_wall, green_eval_pct, t_eval_green_total)

    println("---------------------------------------+---------------+---------+-------------")
    @printf("TOTAL                                  | %13.3f | %6.1f%% | %11.3f\n",
        total_wall, 100.0, t_eval_prop_total + t_eval_green_total)

    println()

    println("========================================")
    println("  QMBToolkitSOEHYB Solver Finished  ")
    println("========================================")
end

function print_startup_message()
    println("========================================")
    println("  QMBToolkitSOEHYB Solver Starting Up  ")
    println("========================================")

    println()
    println("--- Definition of expansion orders ---")
    println("Propagator: number of hybridization lines")
    println("Self-energy: number of hybridization lines")
    println("Green's function: number of hybridization lines + 1")
end
