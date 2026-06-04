# Main control loop
using MPI
using Printf

function solve(solver_data::SolverData, solver_params::SolverParams, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Start total program timer
    t_program_start = MPI.Wtime()

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

    # === Auto-resume from checkpoint (if any) ===
    resume = find_resume_state(solver_data.model, solver_params, comm)
    if rank == 0
        if resume.phase == :none
            println("\nNo checkpoint found; starting fresh.")
        else
            println("\nResuming from checkpoint: phase=$(resume.phase), " *
                    "last completed order=$(resume.last_completed_order).")
        end
    end

    # === Propagator Calculation ===
    if rank == 0
        println()
        println("--- Generate and evaluate propagator ---")
        println("Use bold expansion for the propagator: ", solver_params.propagator.is_bold)
        @boundscheck println("Boundscheck is enabled.")
    end

    if solver_params.propagator.is_bold
        error("Bold propagator calculation (i.e., bold expansion based on self-energy) not yet implemented.")
    end

    t_start = MPI.Wtime()
    eval_diagrams!(solver_data, solver_params, :propagator_diagram, comm; resume = resume)
    t_prop_wall = MPI.Wtime() - t_start

    # On a :green resume the propagator phase is skipped and these accumulators are
    # repopulated later by the green restore, so only print when available.
    if rank == 0 && !isempty(solver_data.calc.partition_function_by_order)
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

    t_start = MPI.Wtime()
    eval_diagrams!(solver_data, solver_params, :green_diagram, comm; resume = resume)
    t_green_wall = MPI.Wtime() - t_start

    # === Data Processing and Saving (rank 0 only) ===
    if rank == 0
        save_results(solver_data, solver_params)
    end

    # The result is now persisted to sim.h5. Barrier makes "all ranks finished"
    # explicit, then rank 0 removes the checkpoint so a fresh rerun starts clean.
    MPI.Barrier(comm)
    if rank == 0
        delete_checkpoint(solver_params.output_path)
    end

    # Record total wall time (before MPI reduction)
    total_wall_time = MPI.Wtime() - t_program_start

    # Compute CPU times across all ranks
    t_prop_cpu = MPI.Reduce(t_prop_wall, MPI.SUM, 0, comm)
    t_green_cpu = MPI.Reduce(t_green_wall, MPI.SUM, 0, comm)
    total_cpu_time = MPI.Reduce(total_wall_time, MPI.SUM, 0, comm)

    # === Print Timing Summary and Save Timing Data (rank 0 only) ===
    if rank == 0
        # Save timing data to file
        save_timing_data(solver_params,
            t_prop_wall, t_prop_cpu,
            t_green_wall, t_green_cpu,
            total_wall_time, total_cpu_time)

        print_shutdown_message(
            nprocs,
            t_prop_wall, t_prop_cpu,
            t_green_wall, t_green_cpu,
            total_wall_time, total_cpu_time
        )
    end
end
