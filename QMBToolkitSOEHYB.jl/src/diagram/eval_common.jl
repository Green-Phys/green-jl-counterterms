### Evaluate the backbone diagrams to obtain their contributions
using MPI
using LinearAlgebra
using QMBToolkitDLR

"""
    eval_diagrams!(solver_data, solver_params, diagram_type, comm)

Evaluate diagrams for a given type using zero-allocation diagram generation.

Diagrams are generated on-the-fly using a single reusable MutableBackBone,
eliminating per-diagram allocation overhead.

# Arguments

  - `solver_data::SolverData{T}`: The solver data
  - `solver_params::SolverParams`: Solver parameters
  - `diagram_type::Symbol`: Type of diagrams (`:propagator_diagram` or `:green_diagram`)
  - `comm::MPI.Comm`: MPI communicator
"""
function eval_diagrams!(solver_data::SolverData{T}, solver_params::SolverParams,
        diagram_type::Symbol, comm::MPI.Comm;
        resume::ResumeState = ResumeState()) where {T}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    propagator = get_propagator(solver_data.calc)

    if diagram_type == :propagator_diagram && !solver_params.propagator.is_bold
        # On a :green resume the propagator phase is already complete; its blocks
        # are not needed (bare green uses the bare propagator) and the green
        # restore repopulates partition_function_by_order. Skip this phase.
        resume.phase == :green && return

        if resume.phase == :propagator
            h5open(checkpoint_path(solver_params.output_path), "r") do file
                restore_checkpoint_propagator!(solver_data, file)
            end
            # Barrier so a fast rank cannot enter the next Allreduce before a slow
            # rank has finished reading the (identical) checkpoint.
            MPI.Barrier(comm)
            start_order = resume.last_completed_order + 1
        else
            empty!(solver_data.calc.partition_function_by_order)
            empty!(solver_data.calc.propagator_diagram_info)
            start_order = 0
        end

        for order in start_order:(solver_params.propagator.expansion_order)
            cache = build_config_cache(order, solver_data.model.n_flavors_physical,
                solver_data.model.hyb.exponents, :propagator_diagram)
            dims = config_dimensions(cache)
            n_total = total_diagrams(cache)

            if rank == 0
                println("Evaluating ", dims, " = ", n_total, " propagator diagrams at order ", order)
            end

            # Store diagram info: (n_topologies, n_directions, n_flavors_combo, n_exponents_combo, total)
            push!(solver_data.calc.propagator_diagram_info, (dims..., n_total))

            eval_propagator_given_order!(solver_data, cache, comm)

            propagator_at_beta = extract_at_tau(
                propagator.vals, solver_data.model.beta, solver_data.model.dlr_grid)
            Z_current = tr(propagator_at_beta)
            # Use real() since partition function must be real; imaginary part is numerical noise
            push!(solver_data.calc.partition_function_by_order, real(Z_current))

            # Checkpoint after each completed order (rank 0, atomic), then the
            # test crash hook (inert in production).
            rank == 0 && write_checkpoint_propagator!(solver_data, solver_params, order, nprocs)
            _maybe_crash(:propagator, order, comm)
        end

        # A boundary resume (start_order == P+1) skips the loop above; this still
        # runs harmlessly on the restored vals (coeffs are unused by bare green).
        propagator.coeffs = tau2dlr(
            solver_data.model.dlr_grid, propagator.vals, solver_data.model.dlr_grid.τ)

    elseif diagram_type == :green_diagram && !solver_params.green.is_bold
        if resume.phase == :green
            h5open(checkpoint_path(solver_params.output_path), "r") do file
                restore_checkpoint_green!(solver_data, file)
            end
            MPI.Barrier(comm)
            start_order = resume.last_completed_order + 1
        else
            empty!(solver_data.calc.green_numerator_by_order)
            empty!(solver_data.calc.green_diagram_info)
            start_order = 1
        end

        for order in start_order:(solver_params.green.expansion_order)
            cache = build_config_cache(order, solver_data.model.n_flavors_physical,
                solver_data.model.hyb.exponents, :green_diagram; is_bold = false)
            dims = config_dimensions(cache)
            n_total = total_diagrams(cache)

            if rank == 0
                println("Evaluating ", dims, " = ", n_total, " Green's function diagrams at order ", order)
            end

            # Store diagram info: (n_topologies, n_directions, n_flavors_combo, n_exponents_combo, total)
            push!(solver_data.calc.green_diagram_info, (dims..., n_total))

            eval_green_given_order!(solver_data, cache, solver_params.green.is_bold, comm)

            # green.vals accumulates the numerator across orders; snapshot it.
            push!(solver_data.calc.green_numerator_by_order,
                copy(solver_data.calc.green.vals))

            rank == 0 && write_checkpoint_green!(solver_data, solver_params, order, nprocs)
            _maybe_crash(:green, order, comm)
        end

        # Rebuild Green's function by order (numerator / Z) from the full per-order
        # vectors; identical on a fresh run and a resumed one.
        empty!(solver_data.calc.green_by_order)
        for green_order in 1:(solver_params.green.expansion_order)
            propagator_index = green_order
            green_index = green_order
            green_at_order_data = solver_data.calc.green_numerator_by_order[green_index] ./
                                  solver_data.calc.partition_function_by_order[propagator_index]
            push!(solver_data.calc.green_by_order, green_at_order_data)
        end

        # Update final Green's function
        copyto!(solver_data.calc.green.vals, solver_data.calc.green_by_order[end])
        solver_data.calc.green.coeffs = QMBToolkitDLR.tau2dlr(
            solver_data.model.dlr_grid, solver_data.calc.green.vals, solver_data.model.dlr_grid.τ; axis = 3)
    else
        error("Unknown or unsupported diagram type: $diagram_type")
    end
end

"""
    compute_diagram_prefactor(diagram::MutableBackBone, K0_inverse::Vector{Float64})

Compute the overall prefactor for a diagram.

The prefactor is:
sign * ∏ᵢ (1/K₀ᵢ)^{Kexp_i}

where:

  - sign is the diagram's sign (from topology and crossing parity)
  - K₀ᵢ are the hybridization function values at τ=0
  - Kexp_i are the exponent counts for each SOE component

Only non-zero exponents contribute to the product.
"""
function compute_diagram_prefactor(diagram::MutableBackBone, K0_inverse::Vector{Float64})
    prefactor = diagram.sign
    for (idx, Kexp) in enumerate(diagram.prefactor_Kexps)
        if Kexp != 0
            prefactor *= K0_inverse[idx]^Kexp
        end
    end
    return prefactor
end
