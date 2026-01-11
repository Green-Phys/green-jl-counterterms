"""
Abstract interface for format-agnostic operations.

This module defines the interface that both Dense and BlockDense formats must implement.
Functions dispatch on propagator/operator types, allowing diagram evaluation code to be
written once and work with any format.

# Interface Functions

## Propagator Operations
- `zeros_like(template)`: Create zero-initialized propagator matching template structure
- `zeros_like_with_flux(template, op)`: Create zero-initialized propagator with flux from operator
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

## Buffer Allocation
- `allocate_propagator_buffers(solver_data)`: Allocate buffers for propagator evaluation
- `allocate_green_buffers(solver_data)`: Allocate buffers for Green's function evaluation

## MPI Operations
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
function tau2dlr! end  # In-place version for buffer reuse
function dlr2tau end
function do_reflection_tau2tau end
function extract_at_tau end
function do_convolution_dlr2tau! end

# Operator Operations
function multiply_op_prop! end
function multiply_prop_op! end
function multiply_prop_prop! end
function trace_prop! end

# Hybridization
function apply_hybridization_inplace! end

# Buffer Allocation
function allocate_propagator_buffers end
function allocate_green_buffers end

# MPI Operations
function mpi_reduce_and_add! end

# Propagator creation
function zeros_like end
function zeros_like_with_flux end  # Create zeros with flux from operator
function add_scaled! end

# Flux handling
function get_flux end
function compute_new_flux end
function compute_new_flux_reverse end

# For Dense mode, flux is always Nothing (zero)
is_zero_flux(::Nothing) = true

# Buffer accessors (used in backbone evaluation)
function get_result end
function get_hyb_temp end
function get_op_product end
function get_result_dlr end
function get_conv_operand end
function get_conv_hyb_temp end
function get_conv_dlr end

# Buffer accessors (used in Green's function evaluation)
function get_contrib_tau_to_beta_reflected end
function get_ann_product end
function get_final_product end

# get_cre_product: generic accessor (cre_product always has zero flux)
get_cre_product(buffers) = buffers.cre_product
