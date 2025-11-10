using TOML
using QMBToolkitDLR
using Printf

"""
    read_input(file_path::String)

Read input parameters from a TOML file.
"""
function read_input(file_path::String)
    dict = TOML.parsefile(file_path)

    @assert dict["model"]["beta"] > 0 "beta must be positive"
    @assert dict["solver"]["propagator"]["expansion_order"] >= 0 "expansion_order must be non-negative"
    @assert dict["solver"]["green"]["expansion_order"] >= 0 "expansion_order must be non-negative"
    @assert dict["solver"]["dlr"]["rtol"] > 0 "rtol must be positive"
    @assert dict["solver"]["dlr"]["Euv"] > 0 "Euv must be positive"

    model_params = ModelParams(
        dict["model"]["beta"],
        dict["model"]["hopping_file"],
        dict["model"]["interaction_file"],
        dict["model"]["hybridization_file"],
        dict["model"]["poles_energies_file"],
        dict["model"]["poles_hybs_file"]
    )

    propagator_params = PropagatorParams(
        dict["solver"]["propagator"]["expansion_order"]
    )

    green_params = GreenParams(
        dict["solver"]["green"]["expansion_order"]
    )

    dlr_params = DLRParams(
        dict["solver"]["dlr"]["Euv"],
        dict["solver"]["dlr"]["rtol"]
    )

    solver_params = SolverParams(
        propagator_params,
        green_params,
        dlr_params
    )

    return Params(model_params, solver_params)
end

"""
    print_parameters(params::Params)

Print parameters in a formatted way.
"""
function print_parameters(params::Params)
    println("="^40)
    println("Input Parameters")
    println("-"^40)
    println(" [model]")
    println("   beta: ", params.model.beta)
    println("   hopping_file: ", params.model.hopping_file)
    println("   interaction_file: ", params.model.interaction_file)
    println("   hybridization_file: ", params.model.hybridization_file)
    println("  [solver]")
    println("   [propagator]")
    println("     expansion_order: ", params.solver.propagator.expansion_order)
    println("   [green]")
    println("     expansion_order: ", params.solver.green.expansion_order)
    println("   [dlr]")
    println("     Euv: ", params.solver.dlr.Euv)
    println("     rtol: ", params.solver.dlr.rtol)
    println("="^40)
end

"""
    print_init_data(solver_data::SolverData)

Print the contents of a `SolverData` object.
"""
function print_init_data(solver_data::SolverData)
    println("="^40)
    println("Model Data")
    println("-"^40)
    println(" poles energies : ", solver_data.model.poles_energies)
    println(" poles hybs : ", solver_data.model.poles_hybs)
    println(" n_flavors: ", solver_data.model.n_flavors)
    println(" n_fock: ", solver_data.model.n_fock)
    println(" Local Hamiltonian Eigenvalues (first 4 out of ", length(solver_data.model.evals), "):")
    if length(solver_data.model.evals) >= 4
        for i in 1:4
            println("  evals[", i, "]: ", solver_data.model.evals[i])
        end
    end
    Z_loc = sum(exp.(-solver_data.model.evals .* solver_data.model.beta))
    Z_poles = compute_Z_noninteracting(solver_data.model.beta, solver_data.model.poles_energies)
    Z_Bethe = compute_Z_Bethe(solver_data.model.beta)^2
    remaining_percentage = 100 * (1 - Z_loc / (Z_poles * Z_Bethe))
    println(" Partition Function for Local Hamiltonian: Z = ", Z_loc)
    println(" Target impurity partition function for Bethe lattice: Z_Bethe = ", Z_Bethe)
    println(" Partition function for poles: Z_poles = ", Z_poles)
    println(" Remaining partition function (%): 1 - Z_loc / (Z_poles * Z_Bethe) = ", @sprintf("%.2f", remaining_percentage), "%")
    println(" Hybridization Function (first 5 tau slices out of ", size(solver_data.model.hyb, 1), "):")
    if size(solver_data.model.hyb, 1) >= 5
        for i in 1:5
            println("  hyb[", i, ", 1, 1]: ", solver_data.model.hyb[i, 1, 1])
        end
    end
    println(" Bare Propagator (first :5 tau slices out of ", size(solver_data.model.bare_propagator, 1), "):")
    if size(solver_data.model.bare_propagator, 1) >= 5
        for i in 1:5
            println("  bare_propagator[", i, ", 1, 1]: ", solver_data.model.bare_propagator[i, 1, 1])
        end
    end
    println("-"^40)
    println("DLR Data")
    println("-"^40)
    println(" # of DLR coefficients: ", solver_data.model.dlr_grid.size)
    println(" DLR Grid Points (first 5):")
    for i in 1:min(5, solver_data.model.dlr_grid.size)
        println("  w[", i, "] = ", solver_data.model.dlr_grid.omega[i])
    end
    println(" # of tau points for DLR grid: ", length(solver_data.model.dlr_grid.tau))
    println(" DLR Tau Points (last 5):")
    for i in max(1, length(solver_data.model.dlr_grid.tau)-4):length(solver_data.model.dlr_grid.tau)
        println("  tau[", i, "] = ", solver_data.model.dlr_grid.tau[i])
    end
    println(" Maximum reconstruction error in hybrdization function: ", solver_data.calc.hyb_error)
    println(" Maximum reconstruction error in bare propagator: ", solver_data.calc.bare_propagator_err)
    println("="^40)
end
