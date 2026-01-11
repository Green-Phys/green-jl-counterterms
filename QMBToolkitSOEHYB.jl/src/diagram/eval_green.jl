# Green's function diagram evaluation for both bare and bold expansion
using MPI

# allocate_green_buffers is defined in format-specific files:
# - src/format/dense/propagator.jl
# - src/format/blockdense/propagator.jl

"""
    eval_green_given_order!(solver_data, diagrams, is_bold, comm)

Evaluate all Green's function diagrams at a given expansion order and accumulate results.
Format-agnostic implementation using interface functions.

# Arguments

  - `solver_data::SolverData{T, Format}`: The solver data containing model information and calculation results
  - `diagrams::Array{BackBone,4}`: 4D array of diagrams organized by (topology, direction, flavor, exponent)
  - `is_bold::Bool`: Whether to use bold expansion (true) or bare expansion (false)
  - `comm::MPI.Comm`: MPI communicator for parallel execution

Updates `solver_data.calc.green.vals` by accumulating contributions from all diagrams.
Uses MPI parallelization to distribute diagram evaluation across ranks.
"""
function eval_green_given_order!(
        solver_data::SolverData{T, Format}, diagrams::Array{BackBone, 4}, is_bold::Bool,
        comm::MPI.Comm) where {T <: Number, Format <: HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = is_bold ? get_propagator(solver_data.calc).vals :
                 get_bare_propagator(solver_data.model).vals

    ann_opts = get_ann_opts(solver_data.model)
    cre_opts = get_cre_opts(solver_data.model)

    local_result = zeros_like(solver_data.calc.green.vals)

    buffers = allocate_green_buffers(solver_data)
    n_tau = length(solver_data.model.dlr_grid.tau)

    diagrams_flat = vcat(diagrams...)
    n_diagrams = length(diagrams_flat)

    my_diagrams = (rank + 1):nprocs:n_diagrams

    for i in my_diagrams
        diagram = diagrams_flat[i]

        contrib_0_to_tau = eval_green_forward_backbone(
            diagram, diagram.external_opts_loc[2], solver_data, propagator, buffers
        )
        contrib_tau_to_beta_unreflected = eval_green_reverse_backbone(
            diagram, diagram.external_opts_loc[2], solver_data, propagator, buffers
        )

        # Reflection: tau -> beta - tau (using pre-allocated buffer)
        flux = get_flux(contrib_tau_to_beta_unreflected)
        contrib_tau_to_beta = get_contrib_tau_to_beta_reflected(buffers, flux)
        do_reflection_tau2tau!(contrib_tau_to_beta, solver_data.model.dlr_grid, contrib_tau_to_beta_unreflected)

        prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)

        for ann_index in 1:solver_data.model.n_flavors_physical
            # Apply annihilation operator: ann_opt * contrib_0_to_tau
            ann_opt = ann_opts[ann_index]
            ann_product_flux = compute_new_flux(get_flux(contrib_0_to_tau), ann_opt)
            ann_product = get_ann_product(buffers, ann_product_flux)
            fill!(ann_product, zero(T))
            multiply_op_prop!(ann_product, ann_opt, contrib_0_to_tau)

            # Multiply: contrib_tau_to_beta * ann_product
            # Result flux = same as backbone_0_to_beta_with_ann[ann_index]
            final_product_flux = get_flux(buffers.backbone_0_to_beta_with_ann[ann_index])
            final_product = get_final_product(buffers, final_product_flux)
            fill!(final_product, zero(T))
            multiply_prop_prop!(final_product, contrib_tau_to_beta, ann_product)

            add_scaled!(buffers.backbone_0_to_beta_with_ann[ann_index], prefactor, final_product)
        end
    end

    # Compute final Green's function by applying creation operators and taking trace
    cre_product = get_cre_product(buffers)
    for ann_index in 1:solver_data.model.n_flavors_physical
        backbone_flux = get_flux(buffers.backbone_0_to_beta_with_ann[ann_index])
        for cre_index in 1:solver_data.model.n_flavors_physical
            cre_opt = cre_opts[cre_index]

            # Skip if total flux is non-zero (trace would be zero anyway)
            total_flux = compute_new_flux_reverse(backbone_flux, cre_opt)
            if !is_zero_flux(total_flux)
                continue
            end

            fill!(cre_product, zero(T))
            multiply_prop_op!(cre_product, buffers.backbone_0_to_beta_with_ann[ann_index], cre_opt)

            trace_prop!(buffers.trace_result_buffer, cre_product)
            view(local_result, :, ann_index, cre_index) .+= buffers.trace_result_buffer
        end
    end

    # Sum results from all ranks for this order, then ADD to existing green values
    mpi_reduce_and_add!(solver_data.calc.green, local_result, comm)
end
