"""
Propagator diagram evaluation for BlockDense representation.

This file contains format-specific evaluation code for BlockDense propagators.
Format-independent diagram generation and evaluation wrapper are in
src/diagram/gen_eval_propagator.jl.
"""

using QMBToolkitED.BlockDense
using QMBToolkitDLR

"""
    PropagatorConvolutionBuffersBlockDense{T}

Buffers for BlockDense propagator backbone convolution calculations.

Since blocks are independent, we store buffers as dictionaries indexed by block pairs.

Fields:
- `basis::BlockDenseBasis`: The basis defining block structure
- `block_buffers::Dict{Tuple{Int,Int}, Tuple{Array{T,3}, Array{T,3}, Array{T,3}}}`:
  Buffers for each block: (operator_product_buffer, backbone_accumulator, convolution_operand_buffer)
"""
struct PropagatorConvolutionBuffersBlockDense{T<:Number}
    basis::BlockDenseBasis
    block_buffers::Dict{Tuple{Int,Int}, Tuple{Array{T,3}, Array{T,3}, Array{T,3}}}
end

"""
    allocate_propagator_buffers(solver_data::SolverData{T, BlockDenseFormat}) where {T<:Number}

Allocate buffers for BlockDense propagator evaluation.

Creates separate buffers for each block in the propagator.

# Arguments
- `solver_data::SolverData{T, BlockDenseFormat}`: Solver data

# Returns
- `PropagatorConvolutionBuffersBlockDense{T}`: Block-specific buffers
"""
function allocate_propagator_buffers(solver_data::SolverData{T, BlockDenseFormat}) where {T<:Number}
    n_tau = length(solver_data.model.dlr_grid.tau)

    # Get basis from bare propagator vals (it's a BlockDenseDLRPropagator, .vals is BlockDensePropagator)
    bare_propagator = get_bare_propagator(solver_data.model)
    bare_prop = bare_propagator.vals
    basis = bare_prop.basis

    block_buffers = Dict{Tuple{Int,Int}, Tuple{Array{T,3}, Array{T,3}, Array{T,3}}}()

    # Allocate buffers for each existing block
    for (block_key, block_array) in bare_prop.blocks
        target_dim = size(block_array, 2)
        source_dim = size(block_array, 3)

        operator_product_buffer = zeros(T, n_tau, target_dim, source_dim)
        backbone_accumulator = zeros(T, n_tau, target_dim, source_dim)
        convolution_operand_buffer = zeros(T, n_tau, target_dim, source_dim)

        block_buffers[block_key] = (operator_product_buffer, backbone_accumulator, convolution_operand_buffer)
    end

    return PropagatorConvolutionBuffersBlockDense{T}(basis, block_buffers)
end

"""
    eval_propagator_backbone(diagram::BackBone, solver_data::SolverData{T, BlockDenseFormat}, buffers::PropagatorConvolutionBuffersBlockDense{T}) where {T<:Number}

Evaluate a single propagator diagram using BlockDense representation.

This is the BlockDense-specific implementation called by the format-independent
`eval_propagator_given_order!` wrapper.

# Arguments
- `diagram::BackBone`: The diagram to evaluate
- `solver_data::SolverData{T, BlockDenseFormat}`: Solver data
- `buffers::PropagatorConvolutionBuffersBlockDense{T}`: Pre-allocated work buffers

# Returns
- `BlockDensePropagator{T}`: Diagram contribution in imaginary time

# Algorithm
Uses `multiplication_convolution_blockdense` to evaluate the backbone diagram via
block-aware convolution in DLR space.
"""
function eval_propagator_backbone(diagram::BackBone, solver_data::SolverData{T, BlockDenseFormat}, buffers::PropagatorConvolutionBuffersBlockDense{T}) where {T<:Number}
    bare_propagator = get_bare_propagator(solver_data.model)
    return multiplication_convolution_blockdense(
        diagram.nodes, diagram.edges,
        solver_data.model.dlr_grid, solver_data.model,
        bare_propagator.vals,
        buffers
    )
end
