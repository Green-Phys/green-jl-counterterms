"""
Order-level checkpoint / restart support.

Long solver runs (high expansion order, large local Hilbert space, MPI on a
cluster) can be killed midway. Because there is no self-consistency loop and each
expansion order is an independent additive contribution that is `Allreduce`-summed
into a per-rank-identical accumulator, we can checkpoint the small accumulator
state after each completed order and resume from it.

Auto-resume is ON by default: rerunning the identical command picks up where it
stopped. The checkpoint lives next to `sim.h5` as `sim.ckpt.h5` and is deleted on
successful completion. Set `QMBTOOLKIT_NO_RESUME=1` to ignore any existing
checkpoint and start fresh.

With the SAME nprocs a resumed run is bit-identical to an uninterrupted one
(per-rank group range and the Allreduce tree are deterministic, and per-block
Allreduce makes Dict iteration order irrelevant). Across DIFFERENT nprocs the
result is still correct but differs at ~1e-16 relative (FP non-associativity).
"""

using MPI
using HDF5

const CHECKPOINT_FORMAT_VERSION = 2

"""
    ResumeState

Resume directive derived from an on-disk checkpoint.

  - `phase ∈ {:none, :propagator, :green}` — which phase the checkpoint stopped in.
  - `last_completed_order` — the highest order FULLY accumulated in that phase
    (-1 when `phase == :none`).
  - `in_flight_groups` — diagram groups of the next order (`last_completed_order+1`)
    already accumulated when the checkpoint was taken mid-order; `0` means the next
    order had not started (an order-boundary checkpoint). Enables within-order
    resume when `checkpoint_group_interval > 0`.
"""
struct ResumeState
    phase::Symbol
    last_completed_order::Int
    in_flight_groups::Int
end
ResumeState() = ResumeState(:none, -1, 0)
ResumeState(phase::Symbol, last_completed_order::Int) = ResumeState(phase, last_completed_order, 0)

"""
    checkpoint_path(output_path)

Checkpoint file path: the `sim.h5` output path with `.h5` → `.ckpt.h5`.
"""
function checkpoint_path(output_path::String)
    out = get_output_file(output_path)
    return replace(out, r"\.h5$" => ".ckpt.h5")
end

# --- Fingerprint ------------------------------------------------------------
# Guards against resuming with a checkpoint produced by an incompatible model /
# parameter set. Values come verbatim from the input files (no arithmetic), so
# exact `==` comparison is appropriate.

"""
    compute_fingerprint(model, params)

Identity fingerprint of the model + solver parameters that a partial result
depends on. `nprocs` is intentionally NOT part of the fingerprint (resume may use
a different rank count); it is stored separately for information only.
"""
function compute_fingerprint(model::ModelData, params::SolverParams)
    basis = get_bare_propagator(model).vals.basis
    sector_dims = Int[s.dim for s in basis.sectors]
    return Dict{String, Any}(
        "beta" => model.beta,
        "prop_order" => params.propagator.expansion_order,
        "green_order" => params.green.expansion_order,
        "prop_is_bold" => Int(params.propagator.is_bold),
        "green_is_bold" => Int(params.green.is_bold),
        "n_flavors" => model.n_flavors,
        "n_flavors_physical" => model.n_flavors_physical,
        "dlr_Euv" => params.dlr.Euv,
        "dlr_rtol" => params.dlr.rtol,
        "exponents" => collect(Float64, model.hyb.exponents),
        "sector_dims" => sector_dims
    )
end

# Navigate a "/"-separated path checking each level, so a missing intermediate
# group returns false rather than erroring across HDF5 versions.
function _haspath(parent, path::AbstractString)
    cur = parent
    parts = split(path, '/')
    for (i, p) in enumerate(parts)
        haskey(cur, String(p)) || return false
        i < length(parts) && (cur = cur[String(p)])
    end
    return true
end

function _write_fingerprint(file, fp::Dict, nprocs::Int)
    grp = create_group(file, "fingerprint")
    for (k, v) in fp
        grp[k] = v
    end
    grp["nprocs_info"] = nprocs
end

function _read_fingerprint(file)
    grp = file["fingerprint"]
    fp = Dict{String, Any}()
    for k in keys(grp)
        k == "nprocs_info" && continue
        fp[k] = read(grp[k])
    end
    return fp
end

"""
    checkpoint_matches(file, expected) -> (matches::Bool, offending_field::String)

Compare the stored fingerprint against `expected`. Returns the first offending
field name on mismatch (for a clear warning), or `("",)` when all fields agree.
"""
function checkpoint_matches(file, expected::Dict)
    stored = _read_fingerprint(file)
    for (k, v) in expected
        haskey(stored, k) || return (false, k)
        stored[k] != v && return (false, k)
    end
    return (true, "")
end

"""
    find_resume_state(model, params, comm) -> ResumeState

Decide whether and how to resume. Every rank reads the checkpoint independently
(the filesystem and environment are identical across ranks), avoiding any
MPI broadcast-API version sensitivity. Warnings are emitted on rank 0 only.

Returns `ResumeState()` (fresh run) when there is no checkpoint, when
`QMBTOOLKIT_NO_RESUME=1`, on a fingerprint mismatch, or on a read failure. A
stale/mismatched checkpoint is left on disk and only overwritten once the new run
checkpoints over it — never deleted before the fresh run succeeds.
"""
function find_resume_state(model::ModelData, params::SolverParams, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    get(ENV, "QMBTOOLKIT_NO_RESUME", "") == "1" && return ResumeState()

    path = checkpoint_path(params.output_path)
    isfile(path) || return ResumeState()

    fp = compute_fingerprint(model, params)
    state = ResumeState()
    try
        h5open(path, "r") do file
            ok, field = checkpoint_matches(file, fp)
            if !ok
                rank == 0 && @warn "Checkpoint fingerprint mismatch (field: $field); starting fresh. The stale checkpoint will be overwritten only after the new run completes." path
                return
            end
            phase = Symbol(read(file["meta/phase"]))
            last_order = Int(read(file["meta/last_completed_order"]))
            in_flight = _haspath(file, "meta/in_flight_groups") ?
                        Int(read(file["meta/in_flight_groups"])) : 0
            state = ResumeState(phase, last_order, in_flight)
        end
    catch err
        rank == 0 && @warn "Failed to read checkpoint; starting fresh." path exception = err
        return ResumeState()
    end
    return state
end

# --- Atomic write -----------------------------------------------------------

# Write to a same-directory temp file, then rename(2) over the target. The
# `h5open` do-block closes the file before `mv`, so the rename sees a complete
# file. Tmp MUST be on the same filesystem as the target (never /tmp) for the
# rename to be atomic. A crash mid-write leaves only the harmless temp file.
function _atomic_write(write_fn::Function, path::String)
    tmp = path * ".tmp." * string(getpid())
    try
        h5open(tmp, "w") do file
            write_fn(file)
        end
        mv(tmp, path; force = true)
    catch err
        isfile(tmp) && rm(tmp; force = true)
        rethrow(err)
    end
    return nothing
end

function _write_meta(file, phase::String, last_order::Int, in_flight_groups::Int)
    grp = create_group(file, "meta")
    grp["phase"] = phase
    grp["last_completed_order"] = last_order
    grp["in_flight_groups"] = in_flight_groups
    grp["format_version"] = CHECKPOINT_FORMAT_VERSION
end

# Diagram info is a Vector{NTuple{5,Int}}; serialize as a 5×n Int matrix exactly
# like save_data (io.jl), so the layout is identical to sim.h5.
function _write_diagram_info(grp, name::String, info::Vector{NTuple{5, Int}})
    isempty(info) || (grp[name] = stack([[t...] for t in info]))
end

function _restore_diagram_info!(dest::Vector{NTuple{5, Int}}, file, path::String)
    empty!(dest)
    if _haspath(file, path)
        mat = read(file[path])
        for j in 1:size(mat, 2)
            push!(dest, (mat[1, j], mat[2, j], mat[3, j], mat[4, j], mat[5, j]))
        end
    end
    return dest
end

# Serialize a block-sparse Propagator's blocks: a sorted Int key matrix [2,n]
# plus one dataset per block. Rebuilt into the Dict on restore.
function _write_blocks(file, group_path::String, blocks::Dict{Tuple{Int, Int}, Array{T, 3}}) where {T}
    grp = create_group(file, group_path)
    ks = sort(collect(keys(blocks)))
    grp["nblocks"] = length(ks)
    if !isempty(ks)
        keymat = Matrix{Int}(undef, 2, length(ks))
        for (c, k) in enumerate(ks)
            keymat[1, c] = k[1]
            keymat[2, c] = k[2]
            grp["block_$(k[1])_$(k[2])"] = blocks[k]
        end
        grp["keys"] = keymat
    end
end

function _restore_blocks!(prop::Propagator{T}, file, group_path::String) where {T}
    grp = file[group_path]
    nblocks = Int(read(grp["nblocks"]))
    empty!(prop.blocks)
    if nblocks > 0
        keymat = read(grp["keys"])
        for c in 1:nblocks
            i, j = Int(keymat[1, c]), Int(keymat[2, c])
            prop.blocks[(i, j)] = convert(Array{T, 3}, read(grp["block_$(i)_$(j)"]))
        end
    end
    return prop
end

# --- Write ------------------------------------------------------------------

"""
    write_checkpoint_propagator!(solver_data, params, last_order, nprocs; in_flight_groups=0)

Rank-0-only. Atomically checkpoint the propagator phase. `last_order` is the
highest FULLY completed order; `in_flight_groups > 0` records a mid-order
checkpoint where that many groups of order `last_order+1` are already accumulated
into the (cumulative) propagator blocks. Stores the partition function and
propagator diagram info for the completed orders plus the accumulated blocks.
"""
function write_checkpoint_propagator!(solver_data::SolverData, params::SolverParams,
        last_order::Int, nprocs::Int; in_flight_groups::Int = 0)
    path = checkpoint_path(params.output_path)
    fp = compute_fingerprint(solver_data.model, params)
    calc = solver_data.calc
    prop = get_propagator(calc)
    _atomic_write(path) do file
        _write_meta(file, "propagator", last_order, in_flight_groups)
        _write_fingerprint(file, fp, nprocs)
        create_group(file, "partition_function")["by_order"] = calc.partition_function_by_order
        _write_diagram_info(create_group(file, "propagator"), "diagram_info",
            calc.propagator_diagram_info)
        _write_blocks(file, "propagator/blocks", prop.vals.blocks)
    end
    return nothing
end

"""
    write_checkpoint_green!(solver_data, params, last_order, nprocs; in_flight_groups=0)

Rank-0-only. Atomically checkpoint the green phase. `last_order` is the highest
FULLY completed green order; `in_flight_groups > 0` records a mid-order checkpoint
where that many groups of order `last_order+1` are already accumulated into the
(cumulative) `green.vals`. Stores the FULL partition function and FULL propagator
diagram info (so the final sim.h5 stays complete) plus the green diagram info, the
accumulated numerator (`green.vals`), and the per-order numerator snapshots
1..last_order.
"""
function write_checkpoint_green!(solver_data::SolverData, params::SolverParams,
        last_order::Int, nprocs::Int; in_flight_groups::Int = 0)
    path = checkpoint_path(params.output_path)
    fp = compute_fingerprint(solver_data.model, params)
    calc = solver_data.calc
    _atomic_write(path) do file
        _write_meta(file, "green", last_order, in_flight_groups)
        _write_fingerprint(file, fp, nprocs)
        create_group(file, "partition_function")["by_order"] = calc.partition_function_by_order
        _write_diagram_info(create_group(file, "propagator"), "diagram_info",
            calc.propagator_diagram_info)
        ggrp = create_group(file, "green")
        _write_diagram_info(ggrp, "diagram_info", calc.green_diagram_info)
        ggrp["vals"] = calc.green.vals
        numgrp = create_group(file, "green/numerator_by_order")
        for (i, arr) in enumerate(calc.green_numerator_by_order)
            numgrp["order_$i"] = arr
        end
    end
    return nothing
end

# --- Restore (all ranks) ----------------------------------------------------

"""
    restore_checkpoint_propagator!(solver_data, file)

Restore propagator-phase accumulators from an open checkpoint `file`: partition
function by order, propagator diagram info, and the propagator value blocks.
"""
function restore_checkpoint_propagator!(solver_data::SolverData, file)
    calc = solver_data.calc
    empty!(calc.partition_function_by_order)
    append!(calc.partition_function_by_order, Float64.(read(file["partition_function/by_order"])))
    _restore_diagram_info!(calc.propagator_diagram_info, file, "propagator/diagram_info")
    _restore_blocks!(get_propagator(calc).vals, file, "propagator/blocks")
    return nothing
end

"""
    restore_checkpoint_green!(solver_data, file)

Restore green-phase state from an open checkpoint `file`: the FULL partition
function, both diagram-info vectors, the per-order numerator snapshots, and the
accumulated `green.vals` (so the next order's `+=` continues correctly).
"""
function restore_checkpoint_green!(solver_data::SolverData, file)
    calc = solver_data.calc
    T = eltype(calc.green.vals)
    empty!(calc.partition_function_by_order)
    append!(calc.partition_function_by_order, Float64.(read(file["partition_function/by_order"])))
    _restore_diagram_info!(calc.propagator_diagram_info, file, "propagator/diagram_info")
    _restore_diagram_info!(calc.green_diagram_info, file, "green/diagram_info")
    empty!(calc.green_numerator_by_order)
    k = 1
    while _haspath(file, "green/numerator_by_order/order_$k")
        push!(calc.green_numerator_by_order,
            convert(Array{T, 3}, read(file["green/numerator_by_order/order_$k"])))
        k += 1
    end
    copyto!(calc.green.vals, convert(Array{T, 3}, read(file["green/vals"])))
    return nothing
end

# --- Cleanup & test hook ----------------------------------------------------

"""
    delete_checkpoint(output_path)

Rank-0-only. Remove the checkpoint and any stale same-directory temp files, so a
fresh rerun starts clean.
"""
function delete_checkpoint(output_path::String)
    path = checkpoint_path(output_path)
    isfile(path) && rm(path; force = true)
    dir, base = dirname(path), basename(path)
    if isdir(dir)
        for f in readdir(dir)
            startswith(f, base * ".tmp.") && rm(joinpath(dir, f); force = true)
        end
    end
    return nothing
end

# Test-only crash hook: when QMBTOOLKIT_CRASH_PHASE / QMBTOOLKIT_CRASH_AFTER_ORDER
# match the current (phase, order), barrier then error on all ranks — modelling a
# realistic mid-run kill AFTER the order's checkpoint is on disk and the HDF5
# handle is closed. Inert unless both env vars are set.
function _maybe_crash(phase::Symbol, order::Int, comm::MPI.Comm)
    crash_phase = get(ENV, "QMBTOOLKIT_CRASH_PHASE", "")
    crash_order = get(ENV, "QMBTOOLKIT_CRASH_AFTER_ORDER", "")
    if crash_phase != "" && crash_order != "" &&
       String(phase) == crash_phase && parse(Int, crash_order) == order
        MPI.Barrier(comm)
        error("QMBTOOLKIT test crash hook: phase=$phase order=$order (checkpoint written)")
    end
    return nothing
end
