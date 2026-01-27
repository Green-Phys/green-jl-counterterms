# Green's function diagram evaluation using batched generation
using MPI

# allocate_green_buffers is defined in format-specific files:
# - src/format/dense/propagator.jl
# - src/format/blockdense/propagator.jl

"""
    eval_green_given_order!(solver_data, cache, is_bold, comm; batch_size=100000)

Evaluate all Green's function diagrams at a given order using batched generation.

Memory-efficient version that generates and evaluates diagrams in batches.

# Arguments

  - `solver_data::SolverData{T, Format}`: Solver data containing model and calculation results
  - `cache::DiagramConfigCache`: Pre-built configuration cache for this order
  - `is_bold::Bool`: Whether to use bold expansion
  - `comm::MPI.Comm`: MPI communicator for parallel evaluation
  - `batch_size::Int`: Number of diagrams per batch (default: 100000)

# Side Effects

  - Updates green by adding contributions from all diagrams

# Parallelization

  - Diagrams are distributed across MPI ranks in contiguous chunks
  - Each rank generates and evaluates its assigned diagrams in batches
  - Results are summed across all ranks using MPI.Allreduce
"""
function eval_green_given_order!(
        solver_data::SolverData{T, Format}, cache::DiagramConfigCache, is_bold::Bool,
        comm::MPI.Comm; batch_size::Int = 100000) where {T <: Number, Format <: HilbertSpaceFormat}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    propagator = is_bold ? get_propagator(solver_data.calc).vals :
                 get_bare_propagator(solver_data.model).vals

    ann_opts = get_ann_opts(solver_data.model)
    cre_opts = get_cre_opts(solver_data.model)

    local_result = zeros_like(solver_data.calc.green.vals)

    # Allocate evaluation buffers once
    buffers = allocate_green_buffers(solver_data, cache.order)

    # Create batch iterator for this rank's portion
    iter = batch_iterator(cache, batch_size, rank, nprocs)

    # Process batches
    for (indices, diagrams) in iter
        for diagram in diagrams
            contrib_0_to_tau = eval_green_forward_backbone(
                diagram, diagram.external_opts_loc[2], solver_data, propagator, buffers
            )
            contrib_tau_to_beta_unreflected = eval_green_reverse_backbone(
                diagram, diagram.external_opts_loc[2], solver_data, propagator, buffers
            )

            # Reflection: tau -> beta - tau (using pre-allocated buffers)
            flux = get_flux(contrib_tau_to_beta_unreflected)
            contrib_tau_to_beta = get_contrib_tau_to_beta_reflected(buffers, flux)
            reflection_dlr = get_reflection_dlr(buffers, flux)
            convolution_workspace = solver_data.model.convolution_workspace
            do_reflection_tau2tau!(contrib_tau_to_beta, reflection_dlr, solver_data.model.dlr_grid,
                contrib_tau_to_beta_unreflected, convolution_workspace.dlr_workspace;
                τ_reflected = convolution_workspace.τ_reflected)

            prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)

            @inbounds for ann_index in 1:solver_data.model.n_flavors_physical
                # Apply annihilation operator: ann_opt * contrib_0_to_tau
                ann_opt = ann_opts[ann_index]
                ann_product_flux = compute_new_flux(get_flux(contrib_0_to_tau), ann_opt)
                ann_product = get_ann_product(buffers, ann_product_flux)
                fill!(ann_product, zero(T))
                multiply_op_prop!(ann_product, ann_opt, contrib_0_to_tau)

                # Multiply: contrib_tau_to_beta * ann_product
                final_product_flux = get_flux(buffers.backbone_0_to_beta_with_ann[ann_index])
                final_product = get_final_product(buffers, final_product_flux)
                fill!(final_product, zero(T))
                multiply_prop_prop!(final_product, contrib_tau_to_beta, ann_product)

                add_scaled!(buffers.backbone_0_to_beta_with_ann[ann_index], prefactor, final_product)
            end
        end
        # Batch is discarded after this iteration, freeing memory
    end

    # Compute final Green's function by applying creation operators and taking trace
    cre_product = get_cre_product(buffers)
    @inbounds for ann_index in 1:solver_data.model.n_flavors_physical
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
            # New layout: (ann_flavor, cre_flavor, n_tau) - tau is now the 3rd dimension
            view(local_result, ann_index, cre_index, :) .+= buffers.trace_result_buffer
        end
    end

    # Sum results from all ranks
    mpi_reduce_and_add!(solver_data.calc.green, local_result, comm)
end
