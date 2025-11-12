# Main control loop
using QMBToolkitDLR
using LinearAlgebra

function solve(solver_data::SolverData, solver_params::SolverParams)
    println("Solver is running with the provided solver data and parameters.")
    calculate_propagator!(solver_data, solver_params)
    calculate_green_function!(solver_data, solver_params)
    uniform_grid = collect(LinRange(0.0, solver_data.model.beta, 1001))
    green_tau = dlr2tau(solver_data.model.dlr_grid, solver_data.calc.green.coeffs, uniform_grid)
    println("Green's function G[1:10, 1,1] at uniform node points:")
    println(green_tau[1:10, 2, 2])
    println("Green's function G[end-9:end, 1,1] at uniform node points:")
    println(green_tau[end-9:end, 2, 2])
    save_green_function(green_tau, uniform_grid, solver_params.output_path*"/sim.h5")
    println("--- Finish calculation ---")
    diagrams = generate_green_diagrams(solver_params.green.expansion_order, solver_data.model.n_flavors, solver_data.model.n_flavors_physical)
    print_diagram(diagrams[1])
end

function calculate_propagator!(solver_data::SolverData, solver_params::SolverParams)
    solver_data.calc.propagator = solver_data.calc.bare_propagator
end

function calculate_green_function!(solver_data::SolverData, solver_params::SolverParams)
    # partition function is the trace of the propagator at tau=beta
    propagator_beta = dlr2tau(solver_data.model.dlr_grid, solver_data.calc.propagator.coeffs, [solver_data.model.beta])
    Z = tr(propagator_beta[1, :, :])
    println("Partition function Z = ", Z)
    # for i in 1:solver_data.model.n_flavors_physical
    #     for j in 1:solver_data.model.n_flavors_physical
    #         # 0th order
    #         c = solver_data.model.ann_opts[i]
    #         cdag = solver_data.model.cre_opts[j]
    #         term1 = Array{Float64,3}(undef, solver_data.model.dlr_grid.size, solver_data.model.n_fock, solver_data.model.n_fock)
    #         for k in 1:solver_data.model.dlr_grid.size
    #             term1[k, :, :] = Matrix(c) * solver_data.calc.propagator_dlr[k, :, :] * Matrix(cdag)
    #         end
    #         term2 = do_reflection_dlr(solver_data.model.dlr_grid, solver_data.calc.propagator_dlr)
    #         term = do_multiplication_dlr(solver_data.model.dlr_grid, term2, term1)
    #         term_trace = [tr(term[l, :, :]) for l in 1:solver_data.model.dlr_grid.size]
    #         solver_data.calc.green_dlr[:, i, j] = -term_trace ./ Z
    #     end
    # end
end