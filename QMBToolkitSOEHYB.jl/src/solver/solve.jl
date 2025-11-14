# Main control loop
using QMBToolkitDLR
using LinearAlgebra

function solve(solver_data::SolverData, solver_params::SolverParams)
    println("Solver is running with the provided solver data and parameters.")
    # gen_diagrams(solver_data, solver_params, :green_diagram)
    diagrams = gen_diagrams(solver_data, solver_params, :propagator_diagram)
    eval_diagrams!(solver_data, solver_params, :propagator_diagram, diagrams)

    ## save data
    uniform_grid = collect(LinRange(0.0, solver_data.model.beta, 1001))
    green_tau = dlr2tau(solver_data.model.dlr_grid, solver_data.calc.green.coeffs, uniform_grid)
    println("Green's function G[1:10, 1,1] at uniform node points:")
    println(green_tau[1:10, 2, 2])
    println("Green's function G[end-9:end, 1,1] at uniform node points:")
    println(green_tau[end-9:end, 2, 2])
    save_green_function(green_tau, uniform_grid, solver_params.output_path*"/sim.h5")
    println("--- Finish calculation ---")
end