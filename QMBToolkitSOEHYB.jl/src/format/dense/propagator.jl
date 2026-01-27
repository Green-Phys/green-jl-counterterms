"""
Backbone buffers and buffer accessors for dense representation.

This file contains:
- DenseBackboneBuffers: Pre-allocated buffers for backbone diagram evaluation
- Flux handling functions (no-ops for dense mode)
- Buffer accessor functions (unified interface matching BlockDense)
- Buffer allocation for propagator and Green's function evaluation
"""

"""
    DenseBackboneBuffers{T}

Buffers for dense backbone diagram evaluation.
Provides 7 distinct buffer roles matching the BlockDense structure for consistency.

All buffers have the same shape since Dense mode doesn't track flux.
The flux parameter in accessor functions is ignored.

# Buffer Roles (tau-sized: n_fock × n_fock × n_tau)

  - `result`: Accumulates the backbone result through multiplication and convolution
  - `hyb_temp`: Scratch buffer for hybridization operations on result
  - `op_product`: Holds operator multiplication output
  - `conv_operand`: Holds propagator copy for convolution
  - `conv_hyb_temp`: Scratch buffer for hybridization on convolution operand

# Buffer Roles (dlr-sized: n_fock × n_fock × n_dlr)

  - `conv_dlr`: DLR transform of convolution operand
  - `result_dlr`: DLR transform of result
"""
struct DenseBackboneBuffers{T <: Number}
    # Tau-sized buffers
    result::DensePropagator{T}
    hyb_temp::DensePropagator{T}
    op_product::DensePropagator{T}
    conv_operand::DensePropagator{T}
    conv_hyb_temp::DensePropagator{T}
    # DLR-sized buffers
    conv_dlr::DensePropagator{T}
    result_dlr::DensePropagator{T}
end

function DenseBackboneBuffers{T}(n_tau::Int, n_fock::Int, n_dlr::Int) where {T <: Number}
    return DenseBackboneBuffers{T}(
        zeros(DensePropagator{T}, n_fock, n_fock, n_tau),   # result
        zeros(DensePropagator{T}, n_fock, n_fock, n_tau),   # hyb_temp
        zeros(DensePropagator{T}, n_fock, n_fock, n_tau),   # op_product
        zeros(DensePropagator{T}, n_fock, n_fock, n_tau),   # conv_operand
        zeros(DensePropagator{T}, n_fock, n_fock, n_tau),   # conv_hyb_temp
        zeros(DensePropagator{T}, n_fock, n_fock, n_dlr),   # conv_dlr
        zeros(DensePropagator{T}, n_fock, n_fock, n_dlr)    # result_dlr
    )
end

#=============================================================================
# Flux Handling (Dense mode doesn't track flux - these are no-ops)
=============================================================================#

"""
    get_flux(prop::DensePropagator)

Get the flux of a Dense propagator. Returns `nothing` since Dense mode doesn't track flux.
"""
get_flux(prop::DensePropagator) = nothing

"""
    compute_new_flux(current_flux::Nothing, op::DenseOperator)

Compute the new flux after operator multiplication for Dense mode.
Returns `nothing` since Dense mode doesn't track flux.
"""
compute_new_flux(current_flux::Nothing, op::DenseOperator) = nothing

"""
    compute_new_flux_reverse(current_flux::Nothing, op::DenseOperator)

Compute the new flux after operator multiplication (reverse direction) for Dense mode.
Returns `nothing` since Dense mode doesn't track flux.
"""
compute_new_flux_reverse(current_flux::Nothing, op::DenseOperator) = nothing

#=============================================================================
# Buffer Accessors (unified interface matching BlockDense)
=============================================================================#

"""
    get_result(buffers::DenseBackboneBuffers{T}, flux) where T

Get the result buffer. Flux parameter is ignored for Dense mode.
"""
get_result(buffers::DenseBackboneBuffers{T}, flux) where {T} = buffers.result

"""
    get_hyb_temp(buffers::DenseBackboneBuffers{T}, flux) where T

Get the hybridization temp buffer. Flux parameter is ignored for Dense mode.
"""
get_hyb_temp(buffers::DenseBackboneBuffers{T}, flux) where {T} = buffers.hyb_temp

"""
    get_op_product(buffers::DenseBackboneBuffers{T}, flux) where T

Get the operator product buffer. Flux parameter is ignored for Dense mode.
"""
get_op_product(buffers::DenseBackboneBuffers{T}, flux) where {T} = buffers.op_product

"""
    get_result_dlr(buffers::DenseBackboneBuffers{T}, flux) where T

Get the result DLR buffer. Flux parameter is ignored for Dense mode.
"""
get_result_dlr(buffers::DenseBackboneBuffers{T}, flux) where {T} = buffers.result_dlr

"""
    get_conv_operand(buffers::DenseBackboneBuffers)

Get the convolution operand buffer.
"""
get_conv_operand(buffers::DenseBackboneBuffers) = buffers.conv_operand

"""
    get_conv_hyb_temp(buffers::DenseBackboneBuffers)

Get the convolution hybridization temp buffer.
"""
get_conv_hyb_temp(buffers::DenseBackboneBuffers) = buffers.conv_hyb_temp

"""
    get_conv_dlr(buffers::DenseBackboneBuffers)

Get the convolution DLR buffer.
"""
get_conv_dlr(buffers::DenseBackboneBuffers) = buffers.conv_dlr

"""
    get_contrib_tau_to_beta_reflected(buffers, flux)

Get the reflection buffer for Green's function evaluation.
Flux parameter is ignored for Dense mode.
"""
get_contrib_tau_to_beta_reflected(buffers, flux) = buffers.contrib_tau_to_beta_reflected

"""
    get_reflection_dlr(buffers, flux)

Get the DLR buffer for reflection operations.
Flux parameter is ignored for Dense mode.
"""
get_reflection_dlr(buffers, flux) = buffers.reflection_dlr

"""
    get_ann_product(buffers, flux)

Get the ann_product buffer. Flux parameter is ignored for Dense mode.
"""
get_ann_product(buffers, flux) = buffers.ann_product

"""
    get_final_product(buffers, flux)

Get the final_product buffer. Flux parameter is ignored for Dense mode.
"""
get_final_product(buffers, flux) = buffers.final_product

#=============================================================================
# Buffer Allocation for Solver
=============================================================================#

"""
    allocate_propagator_buffers(solver_data, expansion_order)

Allocate buffers for dense propagator evaluation.

This is a format-specific function called by the format-independent
`eval_propagator_given_order!` wrapper.

# Arguments

  - `solver_data::SolverData{T}`: Solver data containing grid and Fock space information
  - `expansion_order::Int`: Expansion order (unused in Dense mode, no flux tracking)

# Returns

  - `DenseBackboneBuffers{T}`: Pre-allocated buffers for diagram evaluation
"""
function allocate_propagator_buffers(solver_data::SolverData{T, DenseFormat}, expansion_order::Int) where {T <: Number}
    # Cast to concrete type for type stability
    tau_grid = solver_data.model.dlr_grid.τ::Vector{Float64}
    n_tau = length(tau_grid)
    n_fock = solver_data.model.n_fock
    n_dlr = solver_data.model.dlr_grid.size
    return DenseBackboneBuffers{T}(n_tau, n_fock, n_dlr)
end

"""
    allocate_green_buffers(solver_data, expansion_order)

Allocate evaluation buffers for dense format Green's function computation.

# Arguments

  - `solver_data::SolverData{T, DenseFormat}`: Solver data
  - `expansion_order::Int`: Expansion order (unused in Dense mode, no flux tracking)
"""
function allocate_green_buffers(solver_data::SolverData{T, DenseFormat}, expansion_order::Int) where {T}
    # Cast to concrete type for type stability
    tau_grid = solver_data.model.dlr_grid.τ::Vector{Float64}
    n_tau = length(tau_grid)
    n_fock = solver_data.model.n_fock
    n_dlr = solver_data.model.dlr_grid.size
    n_flavors = solver_data.model.n_flavors_physical

    # Pre-allocate buffers for forward and reverse backbone evaluation
    conv_buffers = (
        fwd_buffers = DenseBackboneBuffers{T}(n_tau, n_fock, n_dlr),
        rev_buffers = DenseBackboneBuffers{T}(n_tau, n_fock, n_dlr)
    )

    # Pre-allocate backbone accumulators (one per annihilation flavor)
    # Explicitly type the Vector for type stability
    backbone_0_to_beta_with_ann = Vector{DensePropagator{T}}(
        [zeros(DensePropagator{T}, n_fock, n_fock, n_tau) for _ in 1:n_flavors]
    )

    # Pre-allocate buffer for reflected contrib_tau_to_beta
    contrib_tau_to_beta_reflected = zeros(DensePropagator{T}, n_fock, n_fock, n_tau)

    # Pre-allocate DLR buffer for reflection operation
    reflection_dlr = zeros(DensePropagator{T}, n_fock, n_fock, n_dlr)

    # Pre-allocate buffers for operator products (reused across loop iterations)
    ann_product = zeros(DensePropagator{T}, n_fock, n_fock, n_tau)
    final_product = zeros(DensePropagator{T}, n_fock, n_fock, n_tau)
    cre_product = zeros(DensePropagator{T}, n_fock, n_fock, n_tau)

    # Pre-allocate trace result buffer (Real type - trace of Hermitian is real)
    trace_buffer = zeros(Float64, n_tau)

    return (
        trace_result_buffer = trace_buffer,
        convolution_buffers = conv_buffers,
        backbone_0_to_beta_with_ann = backbone_0_to_beta_with_ann,
        contrib_tau_to_beta_reflected = contrib_tau_to_beta_reflected,
        reflection_dlr = reflection_dlr,
        ann_product = ann_product,
        final_product = final_product,
        cre_product = cre_product
    )
end
