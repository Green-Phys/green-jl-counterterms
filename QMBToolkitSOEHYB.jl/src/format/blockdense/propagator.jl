"""
Backbone buffers and buffer accessors for BlockDense representation.

This file contains:
- BlockDenseBackboneBuffers: Flux-indexed buffer pools for backbone diagram evaluation
- Flux handling functions (track quantum number changes through operator multiplications)
- Buffer accessor functions (unified interface matching Dense)
- Buffer allocation for propagator and Green's function evaluation

BlockDense mode uses flux-indexed buffer pools because the block structure
depends on the quantum number flux. Each operator multiplication changes the flux,
which determines which blocks exist in the result.

Buffer pools are pre-allocated for all possible flux values based on the maximum
expansion order. The 7 buffer roles match the Dense mode structure for consistency.
"""

using QMBToolkitED.BlockDense

#=============================================================================
# BlockDense Backbone Buffers
=============================================================================#

"""
    BlockDenseBackboneBuffers{T, QN}

Flux-indexed buffer pools for BlockDense backbone diagram evaluation.
Provides 7 distinct buffer roles matching the Dense mode structure for consistency.

Buffer pools are indexed by quantum number (flux), allowing pre-allocation of
buffers for all possible flux values encountered during diagram evaluation.

# Type Parameters

  - `T`: Numeric element type
  - `QN`: Quantum number type (QN_N or QN_NSz)

# Buffer Roles

## Variable-flux pools (indexed by QuantumNumber):

  - `result_pool`: Accumulates the backbone result (flux changes during evaluation)
  - `hyb_temp_pool`: Scratch buffer for hybridization on result (same flux as result)
  - `op_product_pool`: Holds operator multiplication output (new flux after op)
  - `result_dlr_pool`: DLR transform of result (same flux as result)

## Fixed zero-flux buffers:

  - `conv_operand`: Propagator copy for convolution (always zero flux)
  - `conv_hyb_temp`: Scratch buffer for hybridization on conv operand (always zero flux)
  - `conv_dlr`: DLR transform of convolution operand (always zero flux)
"""
struct BlockDenseBackboneBuffers{T <: Number, QN <: QuantumNumber}
    basis::BlockDenseBasis
    n_tau::Int
    n_dlr::Int

    # Variable-flux pools (indexed by QuantumNumber)
    result_pool::Dict{QN, BlockDensePropagator{T}}
    hyb_temp_pool::Dict{QN, BlockDensePropagator{T}}
    op_product_pool::Dict{QN, BlockDensePropagator{T}}
    result_dlr_pool::Dict{QN, BlockDensePropagator{T}}

    # Fixed zero-flux buffers
    zero_flux::OperatorFlux
    conv_operand::BlockDensePropagator{T}
    conv_hyb_temp::BlockDensePropagator{T}
    conv_dlr::BlockDensePropagator{T}
end

"""
    enumerate_possible_flux(::Type{QN_N}, max_order::Int)

Enumerate all possible flux values for QN_N (particle number only).
For expansion order N, flux can range from -N to +N.
"""
function enumerate_possible_flux(::Type{QN_N}, max_order::Int)
    return [QN_N(n) for n in (-max_order):max_order]
end

"""
    enumerate_possible_flux(::Type{QN_NSz}, max_order::Int)

Enumerate all possible flux values for QN_NSz (particle number + spin).
For expansion order N, both n and sz can range from -N to +N,
but only combinations where |sz| <= |n| and (n + sz) is even are valid.
"""
function enumerate_possible_flux(::Type{QN_NSz}, max_order::Int)
    fluxes = QN_NSz[]
    for delta_n in (-max_order):max_order
        for delta_sz in (-max_order):max_order
            # Valid only if |delta_sz| <= |delta_n| and same parity
            if abs(delta_sz) <= abs(delta_n) && (delta_n + delta_sz) % 2 == 0
                push!(fluxes, QN_NSz(delta_n, delta_sz))
            end
        end
    end
    return fluxes
end

"""
    compute_blocks_for_flux(basis::BlockDenseBasis, delta_qn::QN) where QN<:QuantumNumber

Determine which blocks exist for a given flux.
Block (tgt, src) exists if target_qn = source_qn + delta_qn.
"""
function compute_blocks_for_flux(basis::BlockDenseBasis, delta_qn::QN) where {QN <: QuantumNumber}
    blocks = Vector{Tuple{Int, Int}}()
    for (tgt_idx, tgt_sector) in enumerate(basis.sectors)
        for (src_idx, src_sector) in enumerate(basis.sectors)
            # Block exists if target_qn = source_qn + delta_qn
            if tgt_sector.qn == combine_qn(src_sector.qn, delta_qn)
                push!(blocks, (tgt_idx, src_idx))
            end
        end
    end
    return blocks
end

"""
    allocate_buffer_for_flux(basis::BlockDenseBasis, delta_qn::QN, n_time::Int, ::Type{T}) where {T, QN}

Allocate a BlockDensePropagator with the correct block structure for a given flux.
"""
function allocate_buffer_for_flux(basis::BlockDenseBasis, delta_qn::QN, n_time::Int,
        ::Type{T}) where {T <: Number, QN <: QuantumNumber}
    blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    block_indices = compute_blocks_for_flux(basis, delta_qn)

    for (tgt_idx, src_idx) in block_indices
        dim_tgt = basis.sectors[tgt_idx].dim
        dim_src = basis.sectors[src_idx].dim
        blocks[(tgt_idx, src_idx)] = zeros(T, n_time, dim_tgt, dim_src)
    end

    flux = OperatorFlux(delta_qn)
    return BlockDensePropagator{T}(basis, flux, blocks)
end

"""
    BlockDenseBackboneBuffers{T}(basis::BlockDenseBasis, max_order::Int, n_tau::Int, n_dlr::Int) where T

Construct BlockDenseBackboneBuffers with pre-allocated pools for all possible flux values.

# Arguments

  - `basis::BlockDenseBasis`: Basis defining the block structure
  - `max_order::Int`: Maximum expansion order (determines flux range)
  - `n_tau::Int`: Number of tau grid points
  - `n_dlr::Int`: Number of DLR basis functions
"""
function BlockDenseBackboneBuffers{T}(
        basis::BlockDenseBasis, max_order::Int, n_tau::Int, n_dlr::Int) where {T <: Number}
    # Determine quantum number type from basis
    QN = typeof(basis.sectors[1].qn)

    # Enumerate all possible flux values
    possible_fluxes = enumerate_possible_flux(QN, max_order)

    # Pre-allocate variable-flux pools
    result_pool = Dict{QN, BlockDensePropagator{T}}()
    hyb_temp_pool = Dict{QN, BlockDensePropagator{T}}()
    op_product_pool = Dict{QN, BlockDensePropagator{T}}()
    result_dlr_pool = Dict{QN, BlockDensePropagator{T}}()

    for flux_qn in possible_fluxes
        result_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        hyb_temp_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        op_product_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        result_dlr_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_dlr, T)
    end

    # Allocate fixed zero-flux buffers
    zero_qn_val = zero_qn(QN)
    zero_flux = OperatorFlux(zero_qn_val)
    conv_operand = allocate_buffer_for_flux(basis, zero_qn_val, n_tau, T)
    conv_hyb_temp = allocate_buffer_for_flux(basis, zero_qn_val, n_tau, T)
    conv_dlr = allocate_buffer_for_flux(basis, zero_qn_val, n_dlr, T)

    return BlockDenseBackboneBuffers{T, QN}(
        basis, n_tau, n_dlr,
        result_pool, hyb_temp_pool, op_product_pool, result_dlr_pool,
        zero_flux, conv_operand, conv_hyb_temp, conv_dlr
    )
end

#=============================================================================
# Flux Handling
=============================================================================#

"""
    get_flux(prop::BlockDensePropagator)

Get the flux of a BlockDense propagator.
"""
get_flux(prop::BlockDensePropagator) = prop.flux

"""
    compute_new_flux(current_flux::OperatorFlux, op::BlockDenseOperator)

Compute the new flux after operator multiplication for BlockDense mode (forward: op * prop).
"""
function compute_new_flux(current_flux::OperatorFlux, op::BlockDenseOperator)
    OperatorFlux(combine_qn(op.flux.delta_qn, current_flux.delta_qn))
end

"""
    compute_new_flux_reverse(current_flux::OperatorFlux, op::BlockDenseOperator)

Compute the new flux after operator multiplication for BlockDense mode (reverse: prop * op).
"""
function compute_new_flux_reverse(current_flux::OperatorFlux, op::BlockDenseOperator)
    OperatorFlux(combine_qn(current_flux.delta_qn, op.flux.delta_qn))
end

"""
    is_zero_flux(flux::OperatorFlux)

Check if the flux is zero (BlockDense mode).
"""
is_zero_flux(flux::OperatorFlux) = flux.delta_qn == zero_qn(typeof(flux.delta_qn))

#=============================================================================
# Buffer Accessors
=============================================================================#

"""
    get_result(buffers::BlockDenseBackboneBuffers{T,QN}, flux::OperatorFlux) where {T,QN}

Get the result buffer for the given flux.
"""
function get_result(buffers::BlockDenseBackboneBuffers{T, QN}, flux::OperatorFlux) where {T, QN}
    qn = flux.delta_qn::QN
    return buffers.result_pool[qn]
end

"""
    get_hyb_temp(buffers::BlockDenseBackboneBuffers{T,QN}, flux::OperatorFlux) where {T,QN}

Get the hybridization temp buffer for the given flux.
"""
function get_hyb_temp(buffers::BlockDenseBackboneBuffers{T, QN}, flux::OperatorFlux) where {T, QN}
    qn = flux.delta_qn::QN
    return buffers.hyb_temp_pool[qn]
end

"""
    get_op_product(buffers::BlockDenseBackboneBuffers{T,QN}, flux::OperatorFlux) where {T,QN}

Get the operator product buffer for the given flux.
"""
function get_op_product(buffers::BlockDenseBackboneBuffers{T, QN}, flux::OperatorFlux) where {T, QN}
    qn = flux.delta_qn::QN
    return buffers.op_product_pool[qn]
end

"""
    get_result_dlr(buffers::BlockDenseBackboneBuffers{T,QN}, flux::OperatorFlux) where {T,QN}

Get the result DLR buffer for the given flux.
"""
function get_result_dlr(buffers::BlockDenseBackboneBuffers{T, QN}, flux::OperatorFlux) where {T, QN}
    qn = flux.delta_qn::QN
    return buffers.result_dlr_pool[qn]
end

"""
    get_conv_operand(buffers::BlockDenseBackboneBuffers)

Get the convolution operand buffer (always zero flux).
"""
get_conv_operand(buffers::BlockDenseBackboneBuffers) = buffers.conv_operand

"""
    get_conv_hyb_temp(buffers::BlockDenseBackboneBuffers)

Get the convolution hybridization temp buffer (always zero flux).
"""
get_conv_hyb_temp(buffers::BlockDenseBackboneBuffers) = buffers.conv_hyb_temp

"""
    get_conv_dlr(buffers::BlockDenseBackboneBuffers)

Get the convolution DLR buffer (always zero flux).
"""
get_conv_dlr(buffers::BlockDenseBackboneBuffers) = buffers.conv_dlr

"""
    get_contrib_tau_to_beta_reflected(buffers, flux::OperatorFlux)

Get the reflection buffer for the given flux (BlockDense mode).
"""
function get_contrib_tau_to_beta_reflected(buffers, flux::OperatorFlux)
    qn = flux.delta_qn
    return buffers.contrib_tau_to_beta_reflected_pool[qn]
end

"""
    get_ann_product(buffers, flux::OperatorFlux)

Get the ann_product buffer for the given flux (BlockDense mode).
"""
function get_ann_product(buffers, flux::OperatorFlux)
    qn = flux.delta_qn
    return buffers.ann_product_pool[qn]
end

"""
    get_final_product(buffers, flux::OperatorFlux)

Get the final_product buffer for the given flux (BlockDense mode).
"""
function get_final_product(buffers, flux::OperatorFlux)
    qn = flux.delta_qn
    return buffers.final_product_pool[qn]
end

#=============================================================================
# Buffer Allocation for Solver
=============================================================================#

"""
    allocate_propagator_buffers(solver_data, expansion_order)

Allocate buffers for BlockDense propagator evaluation.

# Arguments

  - `solver_data::SolverData{T, BlockDenseFormat}`: Solver data
  - `expansion_order::Int`: Expansion order (determines max flux range for buffer pools)
"""
function allocate_propagator_buffers(solver_data::SolverData{
            T, BlockDenseFormat}, expansion_order::Int) where {T <: Number}
    # Cast to concrete type for type stability
    tau_grid = solver_data.model.dlr_grid.τ
    n_tau = length(tau_grid)
    n_dlr = solver_data.model.dlr_grid.size
    basis = get_bare_propagator(solver_data.model).vals.basis
    # max_order equals expansion_order: maximum flux = ±order
    return BlockDenseBackboneBuffers{T}(basis, expansion_order, n_tau, n_dlr)
end

"""
    allocate_green_buffers(solver_data, expansion_order)

Allocate evaluation buffers for blockdense format Green's function computation.

# Arguments

  - `solver_data::SolverData{T, BlockDenseFormat}`: Solver data
  - `expansion_order::Int`: Expansion order (determines max flux range for buffer pools)
"""
function allocate_green_buffers(solver_data::SolverData{T, BlockDenseFormat}, expansion_order::Int) where {T}
    # Cast to concrete type for type stability
    tau_grid = solver_data.model.dlr_grid.τ
    n_tau = length(tau_grid)
    n_dlr = solver_data.model.dlr_grid.size
    basis = get_bare_propagator(solver_data.model).vals.basis
    n_flavors = solver_data.model.n_flavors_physical

    # max_order equals expansion_order: maximum flux = ±order
    max_order = expansion_order

    # Pre-allocate buffers for forward and reverse backbone evaluation
    conv_buffers = (
        fwd_buffers = BlockDenseBackboneBuffers{T}(basis, max_order, n_tau, n_dlr),
        rev_buffers = BlockDenseBackboneBuffers{T}(basis, max_order, n_tau, n_dlr)
    )

    # Pre-allocate backbone accumulators with correct flux from annihilation operators
    # Explicitly type the Vector for type stability
    ann_opts = get_ann_opts(solver_data.model)
    backbone_0_to_beta_with_ann = Vector{BlockDensePropagator{T}}(
        [zeros(BlockDensePropagator{T}, basis, ann_opts[idx].flux, n_tau) for idx in 1:n_flavors]
    )

    # Pre-allocate flux-indexed pool for reflected contrib_tau_to_beta
    QN = typeof(basis.sectors[1].qn)
    possible_fluxes = enumerate_possible_flux(QN, max_order)
    contrib_tau_to_beta_reflected_pool = Dict{QN, BlockDensePropagator{T}}()
    for flux_qn in possible_fluxes
        contrib_tau_to_beta_reflected_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
    end

    # Pre-allocate ann_product_pool for all possible flux values
    ann_product_pool = Dict{QN, BlockDensePropagator{T}}()
    for flux_qn in possible_fluxes
        ann_product_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
    end

    # Pre-allocate final_product_pool for annihilation operator fluxes only
    final_product_pool = Dict{QN, BlockDensePropagator{T}}()
    for idx in 1:n_flavors
        ann_flux_qn = ann_opts[idx].flux.delta_qn
        if !haskey(final_product_pool, ann_flux_qn)
            final_product_pool[ann_flux_qn] = allocate_buffer_for_flux(basis, ann_flux_qn, n_tau, T)
        end
    end

    # Pre-allocate cre_product with zero flux
    zero_qn_val = zero_qn(QN)
    cre_product = allocate_buffer_for_flux(basis, zero_qn_val, n_tau, T)

    # Pre-allocate trace result buffer (Real type - trace of Hermitian is real)
    trace_buffer = zeros(Float64, n_tau)

    return (
        trace_result_buffer = trace_buffer,
        convolution_buffers = conv_buffers,
        backbone_0_to_beta_with_ann = backbone_0_to_beta_with_ann,
        contrib_tau_to_beta_reflected_pool = contrib_tau_to_beta_reflected_pool,
        ann_product_pool = ann_product_pool,
        final_product_pool = final_product_pool,
        cre_product = cre_product
    )
end
