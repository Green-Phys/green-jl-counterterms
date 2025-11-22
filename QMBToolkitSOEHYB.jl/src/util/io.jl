using TOML
using QMBToolkitDLR
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

    @assert dict["model"]["beta"] > 0 "beta must be positive"
    @assert dict["solver"]["propagator"]["expansion_order"] >= 0 "expansion_order must be non-negative"
    @assert dict["solver"]["green"]["expansion_order"] >= 1 "expansion_order must be at least 1"
    @assert dict["solver"]["dlr"]["rtol"] > 0 "rtol must be positive"
    @assert dict["solver"]["dlr"]["Euv"] > 0 "Euv must be positive"

    # Get the directory containing the input file for resolving relative paths
    input_dir = dirname(abspath(file_path))

    # Resolve paths relative to input file directory
    model_file_path = resolve_path(dict["model"]["file_path"], input_dir)
    output_path = resolve_path(dict["solver"]["output_path"], input_dir)

    # Ensure paths end with directory separator for proper concatenation
    if !endswith(model_file_path, "/")
        model_file_path = model_file_path * "/"
    end
    if !endswith(output_path, "/")
        output_path = output_path * "/"
    end

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

    solver_params = SolverParams(
        propagator_params,
        green_params,
        dlr_params,
        output_path
    )

    return Params(model_params, solver_params)
end

"""
    print_parameters(params::Params)

Print parameters in a formatted way.
"""
function print_parameters(params::Params)
    println("--- Input Parameters ---")
    println(" [model]")
    println("   beta: ", params.model.beta)
    println("   hopping_file: ", params.model.hopping_file)
    println("   interaction_file: ", params.model.interaction_file)
    println("   hybridization_poles_file: ", params.model.hybridization_poles_file)
    println("   hybridization_weights_file: ", params.model.hybridization_weights_file)
    println("  [solver]")
    println("   [expansion]")
    println("     expansion_order: ", params.solver.propagator.expansion_order)
    println("   [green]")
    println("     expansion_order: ", params.solver.green.expansion_order)
    println("   [dlr]")
    println("     Euv: ", params.solver.dlr.Euv)
    println("     rtol: ", params.solver.dlr.rtol)
end

"""
    print_init_data(solver_data::SolverData)

Print the contents of a `SolverData` object.
"""
function print_init_data(solver_data::SolverData)
    println("--- Model Data ---")
    println(" n_flavors: ", solver_data.model.n_flavors)
    println(" n_fock: ", solver_data.model.n_fock)
    println(" n_flavors_physical: ", solver_data.model.n_flavors_physical)
    println(" n_fock_physical: ", solver_data.model.n_fock_physical)
    println(" Local Hamiltonian Eigenvalues (first 4 out of ", length(solver_data.model.evals), "):")
    if length(solver_data.model.evals) >= 4
        for i in 1:4
            println("  evals[", i, "]: ", solver_data.model.evals[i])
        end
    end
    println(" Bare Propagator (first :5 tau slices out of ", size(solver_data.model.bare_propagator.vals, 1), "):")
    if size(solver_data.model.bare_propagator.vals, 1) >= 5
        for i in 1:5
            println("  bare_propagator[", i, ", 1, 1]: ", solver_data.model.bare_propagator.vals[i, 1, 1])
        end
    end
    println("--- DLR Data ---")
    println(" # of DLR coefficients: ", solver_data.model.dlr_grid.size)
    println(" DLR Grid Points (first 5):")
    for i in 1:min(5, solver_data.model.dlr_grid.size)
        println("  w[", i, "] = ", solver_data.model.dlr_grid.omega[i])
    end
    println(" # of tau points for DLR grid: ", length(solver_data.model.dlr_grid.tau))
    println(" DLR Tau Points (last 5):")
    for i in max(1, length(solver_data.model.dlr_grid.tau) - 4):length(solver_data.model.dlr_grid.tau)
        println("  tau[", i, "] = ", solver_data.model.dlr_grid.tau[i])
    end
    println(" Maximum reconstruction error in bare propagator: ", solver_data.model.bare_propagator_err)
end


"""
    save_green_function(green_tau_linear, tau_linear, green_tau_dlr, tau_dlr, green_dlr_coeffs, file_path)

Save Green's function data to an HDF5 file.

Saves both linear grid and DLR grid representations in organized groups:
- /linear_grid/green_tau: Green's function on uniform linear grid
- /linear_grid/tau: Uniform linear tau grid
- /dlr_grid/green_tau: Green's function on DLR tau grid
- /dlr_grid/tau: DLR tau grid points
- /dlr_grid/coeffs: DLR expansion coefficients
"""
function save_green_function(
    green_tau_linear::Array{Float64,3},
    tau_linear::Array{Float64,1},
    green_tau_dlr::Array{Float64,3},
    tau_dlr::Array{Float64,1},
    green_dlr_coeffs::Array{Float64,3},
    file_path::String
)
    h5open(file_path, "w") do file
        # Save linear grid data
        linear_group = create_group(file, "linear_grid")
        linear_group["green_tau"] = green_tau_linear
        linear_group["tau"] = tau_linear

        # Save DLR grid data
        dlr_group = create_group(file, "dlr_grid")
        dlr_group["green_tau"] = green_tau_dlr
        dlr_group["tau"] = tau_dlr
        dlr_group["coeffs"] = green_dlr_coeffs
    end
end
