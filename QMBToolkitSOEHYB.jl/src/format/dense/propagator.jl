"""
Propagator diagram evaluation for dense representation.

This file contains format-specific evaluation code for dense propagators.
Format-independent diagram generation and evaluation wrapper are in
src/diagram/gen_eval_propagator.jl.
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
struct DenseBackboneBuffers{T<:Number}
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

function DenseBackboneBuffers{T}(n_tau::Int, n_fock::Int, n_dlr::Int) where T<:Number
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
function allocate_propagator_buffers(solver_data::SolverData{T, DenseFormat}) where {T<:Number}
    n_tau = length(solver_data.model.dlr_grid.tau)
    n_fock = solver_data.model.n_fock
    n_dlr = solver_data.model.dlr_grid.size  # Number of DLR basis functions
    return DenseBackboneBuffers{T}(n_tau, n_fock, n_dlr)
end


"""
    eval_propagator_backbone(diagram, solver_data, buffers)

Evaluate a single propagator diagram using dense representation.

This is a format-specific function called by the format-independent
`eval_propagator_given_order!` wrapper.

# Arguments
- `diagram::BackBone`: The diagram to evaluate
- `solver_data::SolverData{T, DenseFormat}`: Solver data containing model information
- `buffers::DenseBackboneBuffers{T}`: Pre-allocated work buffers

# Returns
- `DensePropagator{T}`: Diagram contribution in imaginary time (n_tau × n_fock × n_fock)

# Algorithm
Uses `multiplication_convolution` to evaluate the backbone diagram via
convolution in DLR space, which is the core dense-format operation.
"""
function eval_propagator_backbone(diagram::BackBone, solver_data::SolverData{T, DenseFormat}, buffers::DenseBackboneBuffers{T}) where {T<:Number}
    bare_propagator = get_bare_propagator(solver_data.model)
    return multiplication_convolution(
        diagram.nodes, diagram.edges,
        solver_data.model.dlr_grid, solver_data.model,
        bare_propagator.vals,
        buffers
    )
end
