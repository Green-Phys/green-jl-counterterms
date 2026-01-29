"""
Backbone buffers and buffer accessors for diagram evaluation.

This file contains:
- BackboneBuffers: Flux-indexed buffer pools for backbone diagram evaluation
- Flux handling functions (track quantum number changes through operator multiplications)
- Buffer accessor functions (unified interface)
- Buffer allocation for propagator and Green's function evaluation

Buffer pools are pre-allocated for all possible flux values based on the maximum
expansion order. The 7 buffer roles provide scratch space for backbone evaluation.
"""

using QMBToolkitED: Basis, Operator, QuantumNumber, QN_N, QN_NSz

#=============================================================================
# Backbone Buffers
=============================================================================#

"""
    BackboneBuffers{T, QN}

Flux-indexed buffer pools for backbone diagram evaluation.
Provides 7 distinct buffer roles for scratch space during evaluation.

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
struct BackboneBuffers{T <: Number, QN <: QuantumNumber}
    basis::Basis
    n_tau::Int
    n_dlr::Int

    # Variable-flux pools (indexed by QuantumNumber)
    # Only contains entries for fluxes with non-empty block structure
    result_pool::Dict{QN, Propagator{T}}
    hyb_temp_pool::Dict{QN, Propagator{T}}
    op_product_pool::Dict{QN, Propagator{T}}
    result_dlr_pool::Dict{QN, Propagator{T}}

    # Shared empty propagators for fluxes with no valid blocks
    # These are returned for intermediate fluxes that don't exist in the pools
    empty_propagator_tau::Propagator{T}
    empty_propagator_dlr::Propagator{T}

    # Fixed zero-flux buffers
    zero_flux::QN
    conv_operand::Propagator{T}
    conv_hyb_temp::Propagator{T}
    conv_dlr::Propagator{T}
end

"""
    enumerate_possible_flux(::Type{QN_N}, max_order::Int)

Enumerate all possible flux values for QN_N (particle number only).
For expansion order N, flux can range from -N to +N.

Note: This function is deprecated in favor of `compute_valid_fluxes` which
computes only fluxes that have non-empty block structure.
"""
function enumerate_possible_flux(::Type{QN_N}, max_order::Int)
    return [QN_N(n) for n in (-max_order):max_order]
end

"""
    enumerate_possible_flux(::Type{QN_NSz}, max_order::Int)

Enumerate all possible flux values for QN_NSz (particle number + spin).
For expansion order N with 2N operators, both n and sz can range from -2N to +2N.
We include all combinations where (n + sz) is even (parity constraint from operators).

Note: During intermediate steps of diagram evaluation, fluxes outside the final
physical constraint |sz| <= |n| can occur. We must allocate buffers for all
such intermediate fluxes to avoid KeyError during evaluation.

Note: This function is deprecated in favor of `compute_valid_fluxes` which
computes only fluxes that have non-empty block structure.
"""
function enumerate_possible_flux(::Type{QN_NSz}, max_order::Int)
    fluxes = QN_NSz[]
    max_val = 2 * max_order
    for delta_n in (-max_val):max_val
        for delta_sz in (-max_val):max_val
            if (delta_n + delta_sz) % 2 == 0
                push!(fluxes, QN_NSz(delta_n, delta_sz))
            end
        end
    end
    return fluxes
end

"""
    compute_valid_fluxes(basis::Basis)

Compute the set of flux values that have non-empty block structure.
A flux is "valid" here if there exists at least one pair of sectors (tgt, src)
such that tgt.qn = src.qn + flux.

Note: This function is for analysis/benchmarking purposes only. It cannot be used
to optimize buffer allocation because intermediate fluxes during diagram evaluation
may not correspond to any valid sector pair, but still need buffer allocation.
Use `enumerate_possible_flux(QN, max_order)` for buffer allocation instead.

# Returns

  - Vector of flux quantum numbers with non-empty blocks
"""
function compute_valid_fluxes(basis::Basis)
    QN = typeof(basis.sectors[1].qn)
    valid_fluxes = Set{QN}()
    for src_sector in basis.sectors
        for tgt_sector in basis.sectors
            # flux = tgt_qn - src_qn, computed as tgt_qn + (-src_qn)
            flux = tgt_sector.qn + (-src_sector.qn)
            push!(valid_fluxes, flux)
        end
    end
    return collect(valid_fluxes)
end

"""
    compute_blocks_for_flux(basis::Basis, delta_qn::QN) where QN<:QuantumNumber

Determine which blocks exist for a given flux.
Block (tgt, src) exists if target_qn = source_qn + delta_qn.
"""
function compute_blocks_for_flux(basis::Basis, delta_qn::QN) where {QN <: QuantumNumber}
    blocks = Vector{Tuple{Int, Int}}()
    for (tgt_idx, tgt_sector) in enumerate(basis.sectors)
        for (src_idx, src_sector) in enumerate(basis.sectors)
            if tgt_sector.qn == src_sector.qn + delta_qn
                push!(blocks, (tgt_idx, src_idx))
            end
        end
    end
    return blocks
end

"""
    allocate_buffer_for_flux(basis::Basis, delta_qn::QN, n_time::Int, ::Type{T}) where {T, QN}

Allocate a Propagator with the correct block structure for a given flux.
"""
function allocate_buffer_for_flux(basis::Basis, delta_qn::QN, n_time::Int,
        ::Type{T}) where {T <: Number, QN <: QuantumNumber}
    blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()
    block_indices = compute_blocks_for_flux(basis, delta_qn)

    for (tgt_idx, src_idx) in block_indices
        dim_tgt = basis.sectors[tgt_idx].dim
        dim_src = basis.sectors[src_idx].dim
        blocks[(tgt_idx, src_idx)] = zeros(T, dim_tgt, dim_src, n_time)
    end

    return Propagator{T}(basis, delta_qn, blocks)
end

"""
    BackboneBuffers{T}(basis::Basis, max_order::Int, n_tau::Int, n_dlr::Int) where T

Construct BackboneBuffers with pre-allocated pools for all possible flux values.

# Arguments

  - `basis::Basis`: Basis defining the block structure
  - `max_order::Int`: Maximum expansion order (determines flux range)
  - `n_tau::Int`: Number of tau grid points
  - `n_dlr::Int`: Number of DLR basis functions
"""
function BackboneBuffers{T}(
        basis::Basis, max_order::Int, n_tau::Int, n_dlr::Int) where {T <: Number}
    QN = typeof(basis.sectors[1].qn)

    # Use compute_valid_fluxes to only allocate buffers for fluxes with non-empty blocks.
    # Intermediate fluxes during diagram evaluation that have no valid sector pairs
    # will use shared empty propagators (returned by getters for missing keys).
    valid_fluxes = compute_valid_fluxes(basis)

    result_pool = Dict{QN, Propagator{T}}()
    hyb_temp_pool = Dict{QN, Propagator{T}}()
    op_product_pool = Dict{QN, Propagator{T}}()
    result_dlr_pool = Dict{QN, Propagator{T}}()

    for flux_qn in valid_fluxes
        result_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        hyb_temp_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        op_product_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        result_dlr_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_dlr, T)
    end

    # Create shared empty propagators for intermediate fluxes with no valid blocks.
    # These are used for fluxes not in the pools (operations on them produce zero).
    # We use zero(QN) as a placeholder flux - the empty blocks dict is what matters.
    empty_propagator_tau = Propagator{T}(basis, zero(QN), Dict{Tuple{Int, Int}, Array{T, 3}}())
    empty_propagator_dlr = Propagator{T}(basis, zero(QN), Dict{Tuple{Int, Int}, Array{T, 3}}())

    zero_flux = zero(QN)
    conv_operand = allocate_buffer_for_flux(basis, zero_flux, n_tau, T)
    conv_hyb_temp = allocate_buffer_for_flux(basis, zero_flux, n_tau, T)
    conv_dlr = allocate_buffer_for_flux(basis, zero_flux, n_dlr, T)

    return BackboneBuffers{T, QN}(
        basis, n_tau, n_dlr,
        result_pool, hyb_temp_pool, op_product_pool, result_dlr_pool,
        empty_propagator_tau, empty_propagator_dlr,
        zero_flux, conv_operand, conv_hyb_temp, conv_dlr
    )
end

#=============================================================================
# Flux Handling
=============================================================================#

"""
    get_flux(prop::Propagator)

Get the flux of a Propagator.
"""
get_flux(prop::Propagator) = prop.flux

"""
    compute_new_flux(current_flux::QuantumNumber, op::Operator)

Compute the new flux after operator multiplication (forward: op * prop).
"""
function compute_new_flux(current_flux::QuantumNumber, op::Operator)
    op.flux + current_flux
end

"""
    compute_new_flux_reverse(current_flux::QuantumNumber, op::Operator)

Compute the new flux after operator multiplication (reverse: prop * op).
"""
function compute_new_flux_reverse(current_flux::QuantumNumber, op::Operator)
    current_flux + op.flux
end

"""
    is_zero_flux(flux::QuantumNumber)

Check if the flux is zero.
"""
is_zero_flux(flux::QuantumNumber) = flux == zero(typeof(flux))

#=============================================================================
# Buffer Accessors
=============================================================================#

function get_result(buffers::BackboneBuffers{T, QN}, flux::QN) where {T, QN}
    return get(buffers.result_pool, flux, buffers.empty_propagator_tau)
end

function get_hyb_temp(buffers::BackboneBuffers{T, QN}, flux::QN) where {T, QN}
    return get(buffers.hyb_temp_pool, flux, buffers.empty_propagator_tau)
end

function get_op_product(buffers::BackboneBuffers{T, QN}, flux::QN) where {T, QN}
    return get(buffers.op_product_pool, flux, buffers.empty_propagator_tau)
end

function get_result_dlr(buffers::BackboneBuffers{T, QN}, flux::QN) where {T, QN}
    return get(buffers.result_dlr_pool, flux, buffers.empty_propagator_dlr)
end

get_conv_operand(buffers::BackboneBuffers) = buffers.conv_operand
get_conv_hyb_temp(buffers::BackboneBuffers) = buffers.conv_hyb_temp
get_conv_dlr(buffers::BackboneBuffers) = buffers.conv_dlr

function get_contrib_tau_to_beta_reflected(buffers, flux::QuantumNumber)
    return get(buffers.contrib_tau_to_beta_reflected_pool, flux, buffers.empty_propagator_tau)
end

function get_reflection_dlr(buffers, flux::QuantumNumber)
    return get(buffers.reflection_dlr_pool, flux, buffers.empty_propagator_dlr)
end

function get_ann_product(buffers, flux::QuantumNumber)
    return get(buffers.ann_product_pool, flux, buffers.empty_propagator_tau)
end

function get_final_product(buffers, flux::QuantumNumber)
    return get(buffers.final_product_pool, flux, buffers.empty_propagator_tau)
end

#=============================================================================
# Buffer Allocation for Solver
=============================================================================#

"""
    allocate_propagator_buffers(solver_data, expansion_order)

Allocate buffers for propagator evaluation.
"""
function allocate_propagator_buffers(solver_data::SolverData{T}, expansion_order::Int) where {T <: Number}
    tau_grid = solver_data.model.dlr_grid.τ
    n_tau = length(tau_grid)
    n_dlr = solver_data.model.dlr_grid.size
    basis = get_bare_propagator(solver_data.model).vals.basis
    return BackboneBuffers{T}(basis, expansion_order, n_tau, n_dlr)
end

"""
    allocate_green_buffers(solver_data, expansion_order)

Allocate evaluation buffers for Green's function computation.
"""
function allocate_green_buffers(solver_data::SolverData{T}, expansion_order::Int) where {T}
    tau_grid = solver_data.model.dlr_grid.τ
    n_tau = length(tau_grid)
    n_dlr = solver_data.model.dlr_grid.size
    basis = get_bare_propagator(solver_data.model).vals.basis
    n_flavors = solver_data.model.n_flavors_physical

    max_order = expansion_order

    conv_buffers = (
        fwd_buffers = BackboneBuffers{T}(basis, max_order, n_tau, n_dlr),
        rev_buffers = BackboneBuffers{T}(basis, max_order, n_tau, n_dlr)
    )

    ann_opts = get_ann_opts(solver_data.model)
    backbone_0_to_beta_with_ann = Vector{Propagator{T}}(
        [zeros(Propagator{T}, basis, ann_opts[idx].flux, n_tau) for idx in 1:n_flavors]
    )

    QN = typeof(basis.sectors[1].qn)
    # Use compute_valid_fluxes to only allocate buffers for fluxes with non-empty blocks.
    # Missing fluxes will use empty propagators (operations on them produce zero).
    valid_fluxes = compute_valid_fluxes(basis)

    contrib_tau_to_beta_reflected_pool = Dict{QN, Propagator{T}}()
    for flux_qn in valid_fluxes
        contrib_tau_to_beta_reflected_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
    end

    reflection_dlr_pool = Dict{QN, Propagator{T}}()
    for flux_qn in valid_fluxes
        reflection_dlr_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_dlr, T)
    end

    ann_product_pool = Dict{QN, Propagator{T}}()
    for flux_qn in valid_fluxes
        ann_product_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
    end

    # Shared empty propagators for fluxes with no valid blocks
    empty_propagator_tau = Propagator{T}(basis, zero(QN), Dict{Tuple{Int, Int}, Array{T, 3}}())
    empty_propagator_dlr = Propagator{T}(basis, zero(QN), Dict{Tuple{Int, Int}, Array{T, 3}}())

    final_product_pool = Dict{QN, Propagator{T}}()
    for idx in 1:n_flavors
        ann_flux_qn = ann_opts[idx].flux
        if !haskey(final_product_pool, ann_flux_qn)
            final_product_pool[ann_flux_qn] = allocate_buffer_for_flux(basis, ann_flux_qn, n_tau, T)
        end
    end

    cre_product = allocate_buffer_for_flux(basis, zero(QN), n_tau, T)

    trace_buffer = zeros(Float64, n_tau)

    return (
        trace_result_buffer = trace_buffer,
        convolution_buffers = conv_buffers,
        backbone_0_to_beta_with_ann = backbone_0_to_beta_with_ann,
        contrib_tau_to_beta_reflected_pool = contrib_tau_to_beta_reflected_pool,
        reflection_dlr_pool = reflection_dlr_pool,
        ann_product_pool = ann_product_pool,
        final_product_pool = final_product_pool,
        cre_product = cre_product,
        empty_propagator_tau = empty_propagator_tau,
        empty_propagator_dlr = empty_propagator_dlr
    )
end
