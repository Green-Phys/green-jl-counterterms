"""
Backbone buffers and buffer accessors for diagram evaluation.

This file contains:
- PropagatorEvalBuffers: Flux-indexed buffer pools for backbone diagram evaluation
- Flux handling functions (track quantum number changes through operator multiplications)
- Buffer accessor functions (unified interface)
- Buffer allocation for propagator and Green's function evaluation

Buffer pools are pre-allocated for all possible flux values based on the maximum
expansion order. The 5 buffer roles provide scratch space for backbone evaluation.
"""

using QMBToolkitED: Basis, Operator, QuantumNumber, QN_N, QN_NSz

#=============================================================================
# Backbone Buffers
=============================================================================#

"""
    PropagatorEvalBuffers{T, QN}

Flux-indexed buffer pools for backbone diagram evaluation.
Provides 5 distinct buffer roles for scratch space during evaluation.

Buffer pools are indexed by quantum number (flux), allowing pre-allocation of
buffers for all possible flux values encountered during diagram evaluation.

Note: With flux pre-filtering (build_valid_groups), all diagrams are guaranteed
to have valid intermediate fluxes, so no empty propagator sentinels are needed.

# Type Parameters

  - `T`: Numeric element type
  - `QN`: Quantum number type (QN_N or QN_NSz)

# Buffer Roles

## Variable-flux pools (indexed by QuantumNumber):

  - `result_pool`: Accumulates the backbone result (flux changes during evaluation)
  - `op_product_pool`: Holds operator multiplication output (new flux after op)
  - `result_dlr_pool`: DLR transform of result (same flux as result)

## Fixed zero-flux buffers:

  - `conv_operand`: Propagator copy for convolution (always zero flux)
  - `conv_dlr`: DLR transform of convolution operand (always zero flux)
"""
struct PropagatorEvalBuffers{T <: Number, QN <: QuantumNumber}
    basis::Basis
    n_tau::Int
    n_dlr::Int

    # Variable-flux pools (indexed by QuantumNumber)
    # Only contains entries for fluxes with non-empty block structure
    result_pool::Dict{QN, Propagator{T}}
    op_product_pool::Dict{QN, Propagator{T}}
    result_dlr_pool::Dict{QN, Propagator{T}}

    # Fixed zero-flux buffers
    zero_flux::QN
    conv_operand::Propagator{T}
    conv_dlr::Propagator{T}
end

"""
    compute_valid_fluxes(basis::Basis)

Compute the set of flux values that have non-empty block structure.
A flux is "valid" if there exists at least one pair of sectors (tgt, src)
such that tgt.qn = src.qn + flux.

This function is used for buffer allocation. When diagram evaluation encounters
an intermediate flux outside this valid set, it means no valid sector pair exists,
so the propagator has empty blocks and the diagram contributes zero. The backbone
evaluation detects this via `haskey` check and exits early.

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
    PropagatorEvalBuffers{T}(basis::Basis, max_order::Int, n_tau::Int, n_dlr::Int) where T

Construct PropagatorEvalBuffers with pre-allocated pools for all possible flux values.

# Arguments

  - `basis::Basis`: Basis defining the block structure
  - `max_order::Int`: Maximum expansion order (determines flux range)
  - `n_tau::Int`: Number of tau grid points
  - `n_dlr::Int`: Number of DLR basis functions
"""
function PropagatorEvalBuffers{T}(
        basis::Basis, max_order::Int, n_tau::Int, n_dlr::Int) where {T <: Number}
    QN = typeof(basis.sectors[1].qn)

    # Allocate buffers for all valid fluxes (fluxes with non-empty block structure).
    # With flux pre-filtering (build_valid_groups), all diagrams are guaranteed
    # to only encounter valid fluxes during evaluation.
    valid_fluxes = compute_valid_fluxes(basis)

    result_pool = Dict{QN, Propagator{T}}()
    op_product_pool = Dict{QN, Propagator{T}}()
    result_dlr_pool = Dict{QN, Propagator{T}}()

    for flux_qn in valid_fluxes
        result_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        op_product_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        result_dlr_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_dlr, T)
    end

    zero_flux = zero(QN)
    conv_operand = allocate_buffer_for_flux(basis, zero_flux, n_tau, T)
    conv_dlr = allocate_buffer_for_flux(basis, zero_flux, n_dlr, T)

    return PropagatorEvalBuffers{T, QN}(
        basis, n_tau, n_dlr,
        result_pool, op_product_pool, result_dlr_pool,
        zero_flux, conv_operand, conv_dlr
    )
end

#=============================================================================
# Green Evaluation Buffers
=============================================================================#

"""
    GreenEvalBuffers{T, QN}

Buffers for Green's function diagram evaluation.

Contains forward/reverse backbone buffers plus additional pools for
reflection, annihilation/creation operator products, and trace computation.

Note: With flux pre-filtering (build_valid_groups), backbone diagrams are guaranteed
to have valid intermediate fluxes. However, post-backbone operations (ann_product)
may produce fluxes outside the valid set, so empty propagator sentinels are kept
for those accessors.

# Type Parameters

  - `T`: Numeric element type
  - `QN`: Quantum number type (QN_N or QN_NSz)

# Fields

## Backbone evaluation buffers:

  - `fwd_buffers`: PropagatorEvalBuffers for forward backbone (0 → τ)
  - `rev_buffers`: PropagatorEvalBuffers for reverse backbone (τ → β)

## Accumulation buffers:

  - `backbone_0_to_beta_with_ann`: Accumulated backbone with annihilation operator [n_flavors]

## Reflection pools (indexed by flux):

  - `contrib_tau_to_beta_reflected_pool`: τ-space buffer for reflected contribution
  - `reflection_dlr_pool`: DLR-space buffer for reflection transform

## Operator product pools:

  - `ann_product_pool`: Buffer for ann_opt * contrib_0_to_tau (indexed by flux)
  - `final_product_pool`: Buffer for contrib_tau_to_beta * ann_product (selective by ann flux)
  - `cre_product`: Buffer for backbone * cre_opt (zero flux)

## Trace computation:

  - `trace_result_buffer`: Accumulates trace values [n_tau]

## Sentinels:

  - `empty_propagator_tau`: Empty propagator for post-backbone fluxes outside valid set
"""
struct GreenEvalBuffers{T <: Number, QN <: QuantumNumber}
    # Backbone evaluation buffers
    fwd_buffers::PropagatorEvalBuffers{T, QN}
    rev_buffers::PropagatorEvalBuffers{T, QN}

    # Accumulation buffer for each flavor
    backbone_0_to_beta_with_ann::Vector{Propagator{T}}

    # Reflection pools (indexed by flux)
    contrib_tau_to_beta_reflected_pool::Dict{QN, Propagator{T}}
    reflection_dlr_pool::Dict{QN, Propagator{T}}

    # Operator product pools
    ann_product_pool::Dict{QN, Propagator{T}}
    final_product_pool::Dict{QN, Propagator{T}}
    cre_product::Propagator{T}

    # Trace computation
    trace_result_buffer::Vector{Float64}

    # Sentinel for post-backbone fluxes that may be outside valid set
    empty_propagator_tau::Propagator{T}
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

# PropagatorEvalBuffers accessors
# With flux pre-filtering, all accessed fluxes are guaranteed to exist in pools

function get_result(buffers::PropagatorEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.result_pool[flux]
end

function get_op_product(buffers::PropagatorEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.op_product_pool[flux]
end

function get_result_dlr(buffers::PropagatorEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.result_dlr_pool[flux]
end

get_conv_operand(buffers::PropagatorEvalBuffers) = buffers.conv_operand
get_conv_dlr(buffers::PropagatorEvalBuffers) = buffers.conv_dlr

# GreenEvalBuffers accessors
# With flux pre-filtering, all accessed fluxes are guaranteed to exist in pools

function get_contrib_tau_to_beta_reflected(buffers::GreenEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.contrib_tau_to_beta_reflected_pool[flux]
end

function get_reflection_dlr(buffers::GreenEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.reflection_dlr_pool[flux]
end

# Post-backbone accessors use fallback since ann_opt application can produce
# fluxes outside the valid set (ann_product_flux = backbone_flux + ann_opt.flux)
function get_ann_product(buffers::GreenEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return get(buffers.ann_product_pool, flux, buffers.empty_propagator_tau)
end

function get_final_product(buffers::GreenEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return get(buffers.final_product_pool, flux, buffers.empty_propagator_tau)
end

get_cre_product(buffers::GreenEvalBuffers) = buffers.cre_product
get_fwd_buffers(buffers::GreenEvalBuffers) = buffers.fwd_buffers
get_rev_buffers(buffers::GreenEvalBuffers) = buffers.rev_buffers

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
    return PropagatorEvalBuffers{T}(basis, expansion_order, n_tau, n_dlr)
end

"""
    allocate_green_buffers(solver_data, expansion_order)

Allocate evaluation buffers for Green's function computation.

Returns a `GreenEvalBuffers` struct with all pre-allocated pools.
"""
function allocate_green_buffers(solver_data::SolverData{T}, expansion_order::Int) where {T}
    tau_grid = solver_data.model.dlr_grid.τ
    n_tau = length(tau_grid)
    n_dlr = solver_data.model.dlr_grid.size
    basis = get_bare_propagator(solver_data.model).vals.basis
    n_flavors = solver_data.model.n_flavors_physical

    max_order = expansion_order
    QN = typeof(basis.sectors[1].qn)

    # Create forward and reverse backbone buffers
    fwd_buffers = PropagatorEvalBuffers{T}(basis, max_order, n_tau, n_dlr)
    rev_buffers = PropagatorEvalBuffers{T}(basis, max_order, n_tau, n_dlr)

    # Accumulation buffer for each flavor
    ann_opts = get_ann_opts(solver_data.model)
    backbone_0_to_beta_with_ann = Vector{Propagator{T}}(
        [zeros(Propagator{T}, basis, ann_opts[idx].flux, n_tau) for idx in 1:n_flavors]
    )

    # With flux pre-filtering (build_valid_groups), all diagrams have valid fluxes.
    # Allocate buffers for all valid fluxes.
    valid_fluxes = compute_valid_fluxes(basis)

    # Reflection pools
    contrib_tau_to_beta_reflected_pool = Dict{QN, Propagator{T}}()
    reflection_dlr_pool = Dict{QN, Propagator{T}}()
    for flux_qn in valid_fluxes
        contrib_tau_to_beta_reflected_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        reflection_dlr_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_dlr, T)
    end

    # Annihilation product pool
    ann_product_pool = Dict{QN, Propagator{T}}()
    for flux_qn in valid_fluxes
        ann_product_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
    end

    # Final product pool (selective - only allocate for unique ann fluxes)
    final_product_pool = Dict{QN, Propagator{T}}()
    for idx in 1:n_flavors
        ann_flux_qn = ann_opts[idx].flux
        if !haskey(final_product_pool, ann_flux_qn)
            final_product_pool[ann_flux_qn] = allocate_buffer_for_flux(basis, ann_flux_qn, n_tau, T)
        end
    end

    # Single buffers
    cre_product = allocate_buffer_for_flux(basis, zero(QN), n_tau, T)
    trace_buffer = zeros(Float64, n_tau)

    # Empty propagator sentinel for post-backbone fluxes outside valid set
    # Used when ann_opt application produces a flux not in ann_product_pool
    empty_propagator_tau = Propagator{T}(basis, zero(QN), Dict{Tuple{Int, Int}, Array{T, 3}}())

    return GreenEvalBuffers{T, QN}(
        fwd_buffers,
        rev_buffers,
        backbone_0_to_beta_with_ann,
        contrib_tau_to_beta_reflected_pool,
        reflection_dlr_pool,
        ann_product_pool,
        final_product_pool,
        cre_product,
        trace_buffer,
        empty_propagator_tau
    )
end
