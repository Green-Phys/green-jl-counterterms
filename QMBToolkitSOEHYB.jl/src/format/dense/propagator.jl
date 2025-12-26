"""
Propagator diagram evaluation for dense representation.

This file contains format-specific evaluation code for dense propagators.
Format-independent diagram generation and evaluation wrapper are in
src/diagram/gen_eval_propagator.jl.
"""

"""
    PropagatorConvolutionBuffers{T}

Buffers for dense propagator backbone convolution calculations.
Reused across diagrams to avoid allocations during convolution operations.

# Fields
- `operator_product_buffer`: General-purpose buffer for operator multiplications
- `backbone_accumulator`: Accumulates the backbone result through multiplication and convolution
- `convolution_operand_buffer`: Prepares the operand for convolution operations
"""
struct PropagatorConvolutionBuffers{T<:Number}
    operator_product_buffer::DensePropagator{T}
    backbone_accumulator::DensePropagator{T}
    convolution_operand_buffer::DensePropagator{T}
end

function PropagatorConvolutionBuffers{T}(n_tau::Int, n_fock::Int) where T<:Number
    return PropagatorConvolutionBuffers{T}(
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock),
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock),
        zeros(DensePropagator{T}, n_tau, n_fock, n_fock)
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
- `PropagatorConvolutionBuffers{T}`: Pre-allocated buffers for diagram evaluation
"""
function allocate_propagator_buffers(solver_data::SolverData{T, DenseFormat}) where {T<:Number}
    n_tau = length(solver_data.model.dlr_grid.tau)
    n_fock = solver_data.model.n_fock
    return PropagatorConvolutionBuffers{T}(n_tau, n_fock)
end


"""
    eval_propagator_backbone(diagram, solver_data, buffers)

Evaluate a single propagator diagram using dense representation.

This is a format-specific function called by the format-independent
`eval_propagator_given_order!` wrapper.

# Arguments
- `diagram::BackBone`: The diagram to evaluate
- `solver_data::SolverData{T, DenseFormat}`: Solver data containing model information
- `buffers::PropagatorConvolutionBuffers{T}`: Pre-allocated work buffers

# Returns
- `DensePropagator{T}`: Diagram contribution in imaginary time (n_tau × n_fock × n_fock)

# Algorithm
Uses `multiplication_convolution` to evaluate the backbone diagram via
convolution in DLR space, which is the core dense-format operation.
"""
function eval_propagator_backbone(diagram::BackBone, solver_data::SolverData{T, DenseFormat}, buffers::PropagatorConvolutionBuffers{T}) where {T<:Number}
    bare_propagator = get_bare_propagator(solver_data.model)
    return multiplication_convolution(
        diagram.nodes, diagram.edges,
        solver_data.model.dlr_grid, solver_data.model,
        bare_propagator.vals,
        buffers.operator_product_buffer,
        buffers.backbone_accumulator,
        buffers.convolution_operand_buffer
    )
end
