"""
Propagator diagram evaluation for BlockDense representation.

This file contains format-specific evaluation code for BlockDense propagators.
Format-independent diagram generation and evaluation wrapper are in
src/diagram/gen_eval_propagator.jl.

Note: BlockDense mode does not use pre-allocated buffers for convolution operations
because the flux (quantum number change) of intermediate results changes dynamically
during diagram evaluation. Each operator multiplication changes the flux, which
determines which blocks exist in the result. Since we cannot predict this at
buffer allocation time, BlockDense allocates buffers dynamically during evaluation.

The only pre-allocated workspace used is `ConvolutionWorkspace` for DLR convolution,
which is sized to the maximum block dimensions and uses views.
"""

using QMBToolkitED.BlockDense
using QMBToolkitDLR

"""
    eval_propagator_backbone(diagram::BackBone, solver_data::SolverData{T, BlockDenseFormat}) where {T<:Number}

Evaluate a single propagator diagram using BlockDense representation.

This is the BlockDense-specific implementation called by the format-independent
`eval_propagator_given_order!` wrapper.

# Arguments
- `diagram::BackBone`: The diagram to evaluate
- `solver_data::SolverData{T, BlockDenseFormat}`: Solver data

# Returns
- `BlockDensePropagator{T}`: Diagram contribution in imaginary time

# Algorithm
Uses `multiplication_convolution_blockdense` to evaluate the backbone diagram via
block-aware convolution in DLR space.
"""
function eval_propagator_backbone(diagram::BackBone, solver_data::SolverData{T, BlockDenseFormat}) where {T<:Number}
    bare_propagator = get_bare_propagator(solver_data.model)
    return multiplication_convolution_blockdense(
        diagram.nodes, diagram.edges,
        solver_data.model.dlr_grid, solver_data.model,
        bare_propagator.vals
    )
end
