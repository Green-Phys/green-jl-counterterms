"""
Abstract interface for format-agnostic operations.

Functions dispatch on propagator/operator types, allowing diagram evaluation code to be
written once and work with any format.

# Interface Functions

## Propagator Operations
- `zeros_like(template)`: Create zero-initialized propagator matching template structure
- `add_scaled!(dest, scale, src)`: Add scaled contribution: dest += scale * src

## DLR Operations
- `tau2dlr(dlr_grid, prop, tau_grid)`: Transform from tau to DLR representation
- `dlr2tau(dlr_grid, prop_dlr, tau_grid)`: Transform from DLR to tau representation
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
- `allocate_propagator_buffers(solver_data, expansion_order)`: Allocate buffers for propagator evaluation
- `allocate_green_buffers(solver_data, expansion_order)`: Allocate buffers for Green's function evaluation

## MPI Operations
- `mpi_reduce_and_add!(propagator, local_result, comm)`: MPI reduce and add to propagator
"""

using MPI

# ============================================================================
# The following are interface declarations - implementations are in format-specific files
# ============================================================================

# DLR Operations
function tau2dlr end
function tau2dlr! end  # In-place version: tau2dlr!(dest, dlr_grid, src, tau_grid, dlr_workspace)
function dlr2tau end
function do_reflection_tau2tau! end  # In-place reflection F(τ) → F(β-τ)
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
function add_scaled! end

# Flux handling
function get_flux end
function compute_new_flux end
function compute_new_flux_reverse end

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
function get_reflection_dlr end
function get_ann_product end
function get_final_product end

# get_cre_product: generic accessor (cre_product always has zero flux)
get_cre_product(buffers) = buffers.cre_product
