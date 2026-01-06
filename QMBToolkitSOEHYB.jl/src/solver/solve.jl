# Main control loop
using MPI
using Printf

function solve(solver_data::SolverData, solver_params::SolverParams, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Validation
    if (solver_params.propagator.is_bold && solver_params.green.is_bold) ||
       (!solver_params.propagator.is_bold && !solver_params.green.is_bold)
        if rank == 0
            println()
            println("========================================")
            println(" Calculating Diagrams ... ")
            println("========================================")
        end
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

    # === Propagator Calculation ===
    if rank == 0
        println()
        println("--- Generate and evaluate propagator ---")
        println("Use bold expansion for the propagator: ", solver_params.propagator.is_bold)
    end

    if solver_params.propagator.is_bold
        error("Bold propagator calculation not yet implemented.")
    else
        # Generate propagator diagrams
        t_start = MPI.Wtime()
        diagrams_prop = gen_diagrams(solver_data, solver_params, :propagator_diagram, comm)
        t_gen_prop_wall = MPI.Wtime() - t_start

        # Evaluate propagator diagrams
        t_start = MPI.Wtime()
        eval_diagrams!(solver_data, solver_params, :propagator_diagram, diagrams_prop, comm)
        t_eval_prop_wall = MPI.Wtime() - t_start

        # Compute total CPU time across all ranks for evaluation
        t_eval_prop_total = MPI.Reduce(t_eval_prop_wall, MPI.SUM, 0, comm)
    end

    if rank == 0
        println("Partition function Z = ", solver_data.calc.partition_function_by_order[end])
        println("\nPartition function by expansion order:")
        for (idx, Z) in enumerate(solver_data.calc.partition_function_by_order)
            order = idx - 1
            @printf("  Order %d: Z = %.16f\n", order, Z)
        end
    end

    # === Green's Function Calculation ===
    if rank == 0
        println()
        println("--- Generate and evaluate Green's function ---")
        println("Use bold expansion for the Green's function: ", solver_params.green.is_bold)
    end

    # Generate Green's function diagrams
    t_start = MPI.Wtime()
    diagrams_green = gen_diagrams(solver_data, solver_params, :green_diagram, comm)
    t_gen_green_wall = MPI.Wtime() - t_start

    # Evaluate Green's function diagrams
    t_start = MPI.Wtime()
    eval_diagrams!(solver_data, solver_params, :green_diagram, diagrams_green, comm)
    t_eval_green_wall = MPI.Wtime() - t_start

    # Compute total CPU time across all ranks for evaluation
    t_eval_green_total = MPI.Reduce(t_eval_green_wall, MPI.SUM, 0, comm)

    # === Data Processing and Saving (rank 0 only) ===
    if rank == 0
        # Calculate total times
        total_wall_time = t_gen_prop_wall + t_eval_prop_wall + t_gen_green_wall + t_eval_green_wall
        total_cpu_time = t_eval_prop_total + t_eval_green_total

        save_results(solver_data, solver_params, total_wall_time, total_cpu_time)
    end

    # === Print Timing Summary (rank 0 only) ===
    if rank == 0
        print_timing_summary(
            nprocs,
            t_gen_prop_wall, t_eval_prop_wall, t_eval_prop_total,
            t_gen_green_wall, t_eval_green_wall, t_eval_green_total,
        )
    end
end
