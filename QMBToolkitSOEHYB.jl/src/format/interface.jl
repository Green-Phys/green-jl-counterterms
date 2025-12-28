"""
Abstract interface for format-agnostic operations.

This module defines the interface that both Dense and BlockDense formats must implement.
Functions dispatch on propagator/operator types, allowing diagram evaluation code to be
written once and work with any format.

# Interface Functions

## Propagator Operations
- `zeros_like(template)`: Create zero-initialized propagator matching template structure
- `add_scaled!(dest, scale, src)`: Add scaled contribution: dest += scale * src
- `get_data(prop)`: Get underlying data array

## DLR Operations
- `tau2dlr(dlr_grid, prop, tau_grid)`: Transform from tau to DLR representation
- `dlr2tau(dlr_grid, prop_dlr, tau_grid)`: Transform from DLR to tau representation
- `do_reflection_tau2tau(dlr_grid, prop_tau)`: Reflection F(τ) → F(β-τ)
- `extract_at_tau(prop, tau, dlr_grid)`: Extract value at specific tau as dense matrix
- `do_convolution_dlr2tau!(workspace, dlr_grid, cache, F_dlr, G_dlr)`: DLR convolution

## Operator Operations
- `multiply_op_prop!(result, op, prop)`: result = op * prop (2D operator × 3D propagator)
- `multiply_prop_op!(result, prop, op)`: result = prop * op (3D propagator × 2D operator)
- `multiply_prop_prop!(result, A, B)`: result = A * B (3D × 3D propagator multiplication)
- `trace_prop!(result_buffer, prop)`: Trace over spatial indices

## Hybridization
- `apply_hybridization_inplace!(buffer, prop, hyb, model_data; is_reverse)`: Apply hybridization

## Convolution Evaluation
- `multiplication_convolution(nodes, edges, dlr_grid, model_data, propagator, buffers)`: Forward chain
- `multiplication_convolution_reverse(nodes, edges, dlr_grid, model_data, propagator, buffers)`: Reverse chain

## Buffer Allocation
- `allocate_propagator_buffers(solver_data)`: Allocate buffers for propagator evaluation
- `allocate_green_buffers(solver_data)`: Allocate buffers for Green's function evaluation

## MPI Operations
- `create_local_accumulator(solver_data)`: Create format-appropriate local accumulator
- `accumulate_contribution!(accumulator, prefactor, contribution)`: Add contribution to accumulator
- `mpi_reduce_and_add!(propagator, local_result, comm)`: MPI reduce and add to propagator
"""

using MPI

# ============================================================================
# Propagator Operations - these have default implementations that can be overridden
# ============================================================================

"""
    get_data(prop)

Get the underlying data array from a propagator.
For DensePropagator, returns prop.data.
For BlockDensePropagator, returns the blocks dictionary.
"""
function get_data end

# ============================================================================
# The following are interface declarations - implementations are in format-specific files
# ============================================================================

# DLR Operations
function tau2dlr end
function dlr2tau end
function do_reflection_tau2tau end
function extract_at_tau end
function do_convolution_dlr2tau! end

# Operator Operations (get_operator is in diagram/backbone_common.jl)
function multiply_op_prop! end
function multiply_prop_op! end
function multiply_prop_prop! end
function trace_prop! end

# Hybridization
function apply_hybridization_inplace! end

# Convolution Evaluation
function multiplication_convolution end
function multiplication_convolution_reverse end

# Buffer Allocation
function allocate_propagator_buffers end
function allocate_green_buffers end

# MPI Operations
function create_local_accumulator end
function accumulate_contribution! end
function mpi_reduce_and_add! end

# Propagator creation
function zeros_like end
function add_scaled! end

# Product buffer allocation
function allocate_product_buffer end
