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

# Buffer Roles (tau-sized: n_tau × n_fock × n_fock)
- `result`: Accumulates the backbone result through multiplication and convolution
- `hyb_temp`: Scratch buffer for hybridization operations on result
- `op_product`: Holds operator multiplication output
- `conv_operand`: Holds propagator copy for convolution
- `conv_hyb_temp`: Scratch buffer for hybridization on convolution operand

# Buffer Roles (dlr-sized: n_dlr × n_fock × n_fock)
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
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock),   # result
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock),   # hyb_temp
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock),   # op_product
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock),   # conv_operand
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock),   # conv_hyb_temp
        zeros(DensePropagator{T}, n_dlr, n_fock, n_fock),   # conv_dlr
        zeros(DensePropagator{T}, n_dlr, n_fock, n_fock),   # result_dlr
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

#=============================================================================
# Buffer Allocation for Solver
=============================================================================#

"""
    allocate_propagator_buffers(solver_data)

Allocate buffers for dense propagator evaluation.

This is a format-specific function called by the format-independent
`eval_propagator_given_order!` wrapper.

# Arguments
- `solver_data::SolverData{T}`: Solver data containing grid and Fock space information

# Returns
- `DenseBackboneBuffers{T}`: Pre-allocated buffers for diagram evaluation
"""
function allocate_propagator_buffers(solver_data::SolverData{T, DenseFormat}) where {T <: Number}
    n_tau = length(solver_data.model.dlr_grid.tau)
    n_fock = solver_data.model.n_fock
    n_dlr = solver_data.model.dlr_grid.size
    return DenseBackboneBuffers{T}(n_tau, n_fock, n_dlr)
end

"""
    allocate_green_buffers(solver_data::SolverData{T, DenseFormat}) where T

Allocate evaluation buffers for dense format Green's function computation.
"""
function allocate_green_buffers(solver_data::SolverData{T, DenseFormat}) where {T}
    n_tau = length(solver_data.model.dlr_grid.tau)
    n_fock = solver_data.model.n_fock
    n_dlr = solver_data.model.dlr_grid.size

    # Pre-allocate buffers for forward and reverse backbone evaluation
    conv_buffers = (
        fwd_buffers = DenseBackboneBuffers{T}(n_tau, n_fock, n_dlr),
        rev_buffers = DenseBackboneBuffers{T}(n_tau, n_fock, n_dlr),
    )

    return (
        trace_result_buffer = zeros(Float64, n_tau),
        convolution_buffers = conv_buffers,
    )
end
