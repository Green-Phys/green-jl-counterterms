# Main control loop
using QMBToolkitDLR
using LinearAlgebra

function solve(solver_data::SolverData, solver_params::SolverParams)
    # gen_diagrams(solver_data, solver_params, :green_diagram)
    println("--- Generate and evaluate propagator ---")
    diagrams = gen_diagrams(solver_data, solver_params, :propagator_diagram)
    eval_diagrams!(solver_data, solver_params, :propagator_diagram, diagrams)
    println("Partition function Z = ", solver_data.calc.partition_function)
    println("--- Generate and evaluate Green's function ---")
    diagrams_green = gen_diagrams(solver_data, solver_params, :green_diagram)
    eval_diagrams!(solver_data, solver_params, :green_diagram, diagrams_green)


    ## save data
    uniform_grid = collect(LinRange(0.0, solver_data.model.beta, 1001))
    green_tau = dlr2tau(solver_data.model.dlr_grid, solver_data.calc.green.coeffs, uniform_grid)
    println("Green's function G[1:4, 1,1] at uniform node points:")
    println(green_tau[1:4, 1, 1 ])
    println("Head + Tail:")
    println(green_tau[1, 1, 1 ]+green_tau[end, 1, 1 ])
    save_green_function(green_tau, uniform_grid, solver_params.output_path*"/sim.h5")
    println("--- Finish calculation ---")
end