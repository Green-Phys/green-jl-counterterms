"""
Backbone buffers and buffer accessors for diagram evaluation.

This file contains:
- PropagatorEvalBuffers: Flux-indexed buffer pools for backbone diagram evaluation
- Flux handling functions (track quantum number changes through operator multiplications)
- Buffer accessor functions (unified interface)
- Buffer allocation for propagator and Green's function evaluation

Buffer pools are pre-allocated for flux values reachable at the given expansion order.
At order M (M hybridization lines), intermediate fluxes satisfy:
- QN_NSz: |ΔN_up| ≤ M and |ΔN_down| ≤ M, giving at most (2M+1)² fluxes
- QN_N: |ΔN| ≤ M, giving at most (2M+1) fluxes
This avoids allocating buffers for unreachable fluxes, significantly reducing memory.
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

  - `result_pool`: Accumulates the backbone result (flux changes during evaluation;
    operator products are written directly into the new-flux entry)
  - `result_dlr_pool`: DLR transform of result (same flux as result)

## Fixed zero-flux buffers:

  - `conv_operand`: Propagator copy for convolution (always zero flux)
  - `conv_dlr`: DLR transform of convolution operand (always zero flux)

## Convolution-operand DLR cache:

The convolution operand is `bare ⊙ ∏K`, which is flavor-independent and depends
only on the multiset of hybridization indices crossing the edge. When the base
propagator is diagonal (bare expansion), the operand is diagonal, so its DLR
transform is cached by the canonical hyb signature and stored in diagonal form
(`dim × r` per sector) to keep memory bounded.

  - `conv_dlr_cache`: signature → per-block diagonal DLR (only valid when base is diagonal)
  - `conv_sig_scratch`: reusable scratch for building the sorted signature key
"""
struct PropagatorEvalBuffers{T <: Number, QN <: QuantumNumber}
    basis::Basis
    n_tau::Int
    n_dlr::Int

    # Variable-flux pools (indexed by QuantumNumber)
    # Only contains entries for fluxes with non-empty block structure
    result_pool::Dict{QN, Propagator{T}}
    result_dlr_pool::Dict{QN, Propagator{T}}

    # Fixed zero-flux buffers
    zero_flux::QN
    conv_operand::Propagator{T}
    conv_dlr::Propagator{T}

    # Convolution-operand DLR cache (diagonal storage, keyed by hyb signature)
    conv_dlr_cache::Dict{Vector{Int}, Dict{Tuple{Int, Int}, Matrix{T}}}
    conv_sig_scratch::Vector{Int}
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
    is_flux_within_order(flux::QN_NSz, max_order::Int)

Check if a QN_NSz flux is reachable at the given expansion order.

At order M (M hybridization lines), each operator changes (N_up, N_down) by (±1, 0) or (0, ±1).
With M creation and M annihilation operators, intermediate fluxes satisfy:
|ΔN_up| ≤ M and |ΔN_down| ≤ M, where ΔN_up = (n + sz) ÷ 2 and ΔN_down = (n - sz) ÷ 2.
"""
function is_flux_within_order(flux::QN_NSz, max_order::Int)
    delta_n_up = (flux.n + flux.sz) ÷ 2
    delta_n_down = (flux.n - flux.sz) ÷ 2
    return abs(delta_n_up) <= max_order && abs(delta_n_down) <= max_order
end

"""
    is_flux_within_order(flux::QN_N, max_order::Int)

Check if a QN_N flux is reachable at the given expansion order.

At order M, with M creation and M annihilation operators, intermediate |ΔN| ≤ M.
"""
function is_flux_within_order(flux::QN_N, max_order::Int)
    return abs(flux.n) <= max_order
end

# Fallback for other quantum number types: no filtering
is_flux_within_order(flux::QuantumNumber, max_order::Int) = true

"""
    compute_order_bounded_fluxes(basis::Basis, max_order::Int)

Compute fluxes that are both structurally valid (non-empty blocks) and reachable
at the given expansion order.

Returns the intersection of `compute_valid_fluxes(basis)` and the order-based bound.
"""
function compute_order_bounded_fluxes(basis::Basis, max_order::Int)
    all_valid = compute_valid_fluxes(basis)
    return filter(flux -> is_flux_within_order(flux, max_order), all_valid)
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

    # Allocate buffers only for fluxes reachable at the given expansion order.
    # At order M, intermediate fluxes satisfy |ΔN_up| ≤ M, |ΔN_down| ≤ M (QN_NSz)
    # or |ΔN| ≤ M (QN_N). This avoids allocating buffers for unreachable fluxes.
    valid_fluxes = compute_order_bounded_fluxes(basis, max_order)

    result_pool = Dict{QN, Propagator{T}}()
    result_dlr_pool = Dict{QN, Propagator{T}}()

    for flux_qn in valid_fluxes
        result_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
        result_dlr_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_dlr, T)
    end

    zero_flux = zero(QN)
    conv_operand = allocate_buffer_for_flux(basis, zero_flux, n_tau, T)
    conv_dlr = allocate_buffer_for_flux(basis, zero_flux, n_dlr, T)

    conv_dlr_cache = Dict{Vector{Int}, Dict{Tuple{Int, Int}, Matrix{T}}}()
    conv_sig_scratch = Int[]

    return PropagatorEvalBuffers{T, QN}(
        basis, n_tau, n_dlr,
        result_pool, result_dlr_pool,
        zero_flux, conv_operand, conv_dlr,
        conv_dlr_cache, conv_sig_scratch
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

Note: With flux pre-filtering (build_valid_groups), all diagrams are guaranteed
to have valid intermediate fluxes. The pre-filtering checks both backbone fluxes
AND external operator fluxes, ensuring at least one valid (α, β) combination exists.
Invalid (α) combinations during evaluation are skipped via haskey check.

# Type Parameters

  - `T`: Numeric element type
  - `QN`: Quantum number type (QN_N or QN_NSz)

# Fields

## Backbone evaluation buffers:

  - `fwd_buffers`: PropagatorEvalBuffers for forward backbone (0 → τ)
  - `rev_buffers`: PropagatorEvalBuffers for reverse backbone (τ → β)

## Accumulation buffers:

  - `backbone_0_to_beta_with_ann`: Accumulated backbone with annihilation operator [n_flavors_physical]

## Reflection pool (indexed by flux):

  - `contrib_tau_to_beta_reflected_pool`: τ-space buffer for reflected contribution

## Operator product pools:

  - `ann_product_pool`: Buffer for ann_opt * contrib_0_to_tau (indexed by flux)
  - `final_product_pool`: Buffer for contrib_tau_to_beta * ann_product (selective by ann flux)
  - `cre_product`: Buffer for backbone * cre_opt (zero flux)

## Trace computation:

  - `trace_result_buffer`: Accumulates trace values [n_tau]
"""
struct GreenEvalBuffers{T <: Number, QN <: QuantumNumber}
    # Backbone evaluation buffers
    fwd_buffers::PropagatorEvalBuffers{T, QN}
    rev_buffers::PropagatorEvalBuffers{T, QN}

    # Accumulation buffer for each flavor
    backbone_0_to_beta_with_ann::Vector{Propagator{T}}

    # Reflection pool (indexed by flux)
    contrib_tau_to_beta_reflected_pool::Dict{QN, Propagator{T}}

    # Operator product pools
    ann_product_pool::Dict{QN, Propagator{T}}
    final_product_pool::Dict{QN, Propagator{T}}
    cre_product::Propagator{T}

    # Trace computation
    trace_result_buffer::Vector{Float64}
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

function get_result_dlr(buffers::PropagatorEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.result_dlr_pool[flux]
end

get_conv_operand(buffers::PropagatorEvalBuffers) = buffers.conv_operand
get_conv_dlr(buffers::PropagatorEvalBuffers) = buffers.conv_dlr

#=============================================================================
# Convolution-operand DLR cache helpers
=============================================================================#

"""
    conv_signature!(scratch::Vector{Int}, attached_hybs)

Build a canonical (sorted) signature of the hybridization indices crossing an
edge into the reusable `scratch` vector and return it. Sorting makes the key
order-independent; the empty edge yields an empty signature.
"""
function conv_signature!(scratch::Vector{Int}, attached_hybs)
    n = length(attached_hybs)
    resize!(scratch, n)
    @inbounds for j in 1:n
        scratch[j] = attached_hybs[j].index
    end
    sort!(scratch)
    return scratch
end

"""
    cache_conv_diagonals(conv_dlr::Propagator{T}) where {T}

Extract the per-block diagonal of a diagonal convolution-operand DLR propagator
into compact `dim × r` matrices. Used to populate the cache on a miss.
"""
function cache_conv_diagonals(conv_dlr::Propagator{T}) where {T}
    cached = Dict{Tuple{Int, Int}, Matrix{T}}()
    for (key, block) in conv_dlr.blocks
        d = size(block, 1)
        r = size(block, 3)
        D = Matrix{T}(undef, d, r)
        @inbounds for k in 1:r
            for a in 1:d
                D[a, k] = block[a, a, k]
            end
        end
        cached[key] = D
    end
    return cached
end

# GreenEvalBuffers accessors
# With flux pre-filtering, all accessed fluxes are guaranteed to exist in pools

function get_contrib_tau_to_beta_reflected(buffers::GreenEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.contrib_tau_to_beta_reflected_pool[flux]
end

# With flux pre-filtering including external operators, all accessed fluxes are valid
function get_ann_product(buffers::GreenEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.ann_product_pool[flux]
end

function get_final_product(buffers::GreenEvalBuffers{T, QN}, flux::QN) where {T, QN}
    return buffers.final_product_pool[flux]
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

    QN = typeof(basis.sectors[1].qn)

    # Green order K = expansion_order. Backbone has K-1 hybridization lines.
    # Forward/reverse backbone intermediate fluxes: |ΔN_up| ≤ K-1, |ΔN_down| ≤ K-1
    backbone_max_order = expansion_order - 1
    fwd_buffers = PropagatorEvalBuffers{T}(basis, backbone_max_order, n_tau, n_dlr)

    # The reverse sweep runs only after the forward sweep has fully completed and
    # parked its result in fwd_buffers.result_pool (still read in the ann loop below).
    # Every other forward buffer is pure per-edge scratch that is dead by then, so the
    # reverse sweep shares them — only result_pool must stay private to avoid clobbering
    # the forward result. The conv-operand DLR cache is keyed by hyb signature and built
    # direction-independently (build_conv_dlr! always uses is_reverse=false), so sharing it
    # is correct and lets the reverse sweep reuse entries the forward sweep already built.
    rev_result_pool = Dict{QN, Propagator{T}}()
    for flux_qn in compute_order_bounded_fluxes(basis, backbone_max_order)
        rev_result_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
    end
    rev_buffers = PropagatorEvalBuffers{T, QN}(
        basis, n_tau, n_dlr,
        rev_result_pool, fwd_buffers.result_dlr_pool,
        fwd_buffers.zero_flux, fwd_buffers.conv_operand, fwd_buffers.conv_dlr,
        fwd_buffers.conv_dlr_cache, fwd_buffers.conv_sig_scratch
    )

    # Accumulation buffer for each flavor
    ann_opts = get_ann_opts(solver_data.model)
    backbone_0_to_beta_with_ann = Vector{Propagator{T}}(
        [zeros(Propagator{T}, basis, ann_opts[idx].flux, n_tau) for idx in 1:n_flavors]
    )

    # Reflection pool: stores reflected reverse backbone result.
    # Same flux range as reverse backbone: order = K-1.
    reflected_fluxes = compute_order_bounded_fluxes(basis, backbone_max_order)
    contrib_tau_to_beta_reflected_pool = Dict{QN, Propagator{T}}()
    for flux_qn in reflected_fluxes
        contrib_tau_to_beta_reflected_pool[flux_qn] = allocate_buffer_for_flux(basis, flux_qn, n_tau, T)
    end

    # Annihilation product pool: stores ann_opt * fwd_backbone_result.
    # Flux = fwd_backbone_flux + ann_op_flux. Ann op adds ±1 to ΔN_up or ΔN_down,
    # so max flux order = (K-1) + 1 = K.
    ann_product_fluxes = compute_order_bounded_fluxes(basis, expansion_order)
    ann_product_pool = Dict{QN, Propagator{T}}()
    for flux_qn in ann_product_fluxes
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

    return GreenEvalBuffers{T, QN}(
        fwd_buffers,
        rev_buffers,
        backbone_0_to_beta_with_ann,
        contrib_tau_to_beta_reflected_pool,
        ann_product_pool,
        final_product_pool,
        cre_product,
        trace_buffer
    )
end
