# Main control loop
using QMBToolkitDLR
using LinearAlgebra

function solve(solver_data::SolverData, solver_params::SolverParams)
    # gen_diagrams(solver_data, solver_params, :green_diagram)
    if solver_params.propagator.is_bold && solver_params.green.is_bold || !solver_params.propagator.is_bold && !solver_params.green.is_bold
        println("--- Start calculation ---")
    else
        error("Invalid combination of boldness settings:" *
              " propagator.is_bold=$(solver_params.propagator.is_bold), " *
              "green.is_bold=$(solver_params.green.is_bold).")
    end

    if solver_params.propagator.expansion_order != solver_params.green.expansion_order-1
        error("For non-bold expansion, propagator expansion order must be one less than green expansion order." *
              " Got propagator.expansion_order=$(solver_params.propagator.expansion_order), " *
              "green.expansion_order=$(solver_params.green.expansion_order).")
    end

    println("--- Generate and evaluate propagator ---")
    println("Use bold expansion for the propagator: ", solver_params.propagator.is_bold)
    if solver_params.propagator.is_bold
        error("Bold propagator calculation not yet implemented.")
    else
        diagrams = gen_diagrams(solver_data, solver_params, :propagator_diagram)
        eval_diagrams!(solver_data, solver_params, :propagator_diagram, diagrams)
    end
    println("Partition function Z = ", solver_data.calc.partition_function)
    println("--- Generate and evaluate Green's function ---")
    println("Use bold expansion for the Green's function: ", solver_params.green.is_bold)
    diagrams_green = gen_diagrams(solver_data, solver_params, :green_diagram)
    eval_diagrams!(solver_data, solver_params, :green_diagram, diagrams_green)


    ## Save data
    # Linear grid
    uniform_grid = collect(LinRange(0.0, solver_data.model.beta, solver_params.green.save_linear_size))
    green_tau_linear = dlr2tau(solver_data.model.dlr_grid, solver_data.calc.green.coeffs, uniform_grid)

    # DLR grid
    dlr_tau_grid = solver_data.model.dlr_grid.tau
    green_tau_dlr = solver_data.calc.green.vals
    green_dlr_coeffs = solver_data.calc.green.coeffs

    # Print summary
    println("First 4 linear grid tau points:")
    println(uniform_grid[1:4])
    println("Green's function G[1:4, 1,1] on linear grid:")
    println(green_tau_linear[1:4, 1, 1])
    println("Head + Tail (linear grid):")
    println(green_tau_linear[1, 1, 1] + green_tau_linear[end, 1, 1])

    # Save both linear and DLR grid data
    save_green_function(
        green_tau_linear, uniform_grid,
        green_tau_dlr, dlr_tau_grid, green_dlr_coeffs,
        solver_params.output_path * "sim.h5"
    )
    println("Saved Green's function (linear grid and DLR grid) to: ", solver_params.output_path * "sim.h5")
    println("--- Finish calculation ---")
end