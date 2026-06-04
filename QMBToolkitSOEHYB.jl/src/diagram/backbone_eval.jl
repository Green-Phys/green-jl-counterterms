"""
Unified backbone diagram evaluation.

Format-agnostic implementation of backbone diagram evaluation that works through
interface functions.

# Key Functions
- `get_operator(node, model_data)`: Get operator for a diagram node
- `multiplication_convolution(...)`: Forward backbone evaluation (right to left)
- `multiplication_convolution_reverse(...)`: Reverse backbone evaluation (left to right)
- `eval_propagator_forward_backbone(...)`: Propagator diagram evaluation
- `eval_green_forward_backbone(...)`: Forward Green's function backbone
- `eval_green_reverse_backbone(...)`: Reverse Green's function backbone

# Interface Functions Used
- get_flux, compute_new_flux, compute_new_flux_reverse: Flux handling
- get_result, get_op_product, get_result_dlr, get_conv_operand, get_conv_dlr: Buffer accessors
- multiply_op_prop!, multiply_prop_op!: Operator multiplications
- apply_hybridization_inplace!: Hybridization application
- tau2dlr!, do_convolution_dlr2tau!: DLR operations
"""

using QMBToolkitDLR

#=============================================================================
# Operator Lookup
=============================================================================#

"""
    get_operator(node::BackBoneNode, model_data::ModelData)

Get the operator for a given node from model data.

This is a format-agnostic function that uses interface accessors to retrieve
the appropriate operator based on the model_data type.

# Arguments

  - `node::BackBoneNode`: Node containing operator type and indices
  - `model_data::ModelData`: Model data with operators

# Returns

Appropriate operator based on node type:

  - `:cdag` with bar_index=-1: Creation operator from exact diagonalization
  - `:c` with bar_index=-1: Annihilation operator from exact diagonalization
  - `:cdag` with bar_index≠-1: SOE-transformed creation operator
  - `:c` with bar_index≠-1: SOE-transformed annihilation operator
"""
function get_operator(node::BackBoneNode, model_data::ModelData)
    if node.opt_type == :cdag
        if node.bar_index == -1
            return get_cre_opts(model_data)[node.flavor]
        else
            return get_cre_opts_bar(model_data)[node.flavor, node.bar_index]
        end
    else  # node.opt_type == :c
        if node.bar_index == -1
            return get_ann_opts(model_data)[node.flavor]
        else
            return get_ann_opts_bar(model_data)[node.flavor, node.bar_index]
        end
    end
end

#=============================================================================
# Core Backbone Convolution Operations
=============================================================================#

"""
    build_conv_dlr!(conv_dlr, conv_operand, propagator, edge, model_data, dlr_grid, dlr_workspace)

Build the convolution-operand DLR transform from scratch: copy the base propagator,
apply the edge's hybridization K-factors, and transform to DLR. Writes into `conv_dlr`.
"""
function build_conv_dlr!(conv_dlr, conv_operand, propagator, edge,
        model_data::ModelData, dlr_grid::DLRGrid, dlr_workspace)
    copyto!(conv_operand, propagator)
    for hyb in get_attached_hybs(edge)
        apply_hybridization_inplace!(conv_operand, hyb, model_data, is_reverse = false)
    end
    tau2dlr!(conv_dlr, dlr_grid, conv_operand, dlr_grid.τ, dlr_workspace)
    return conv_dlr
end

"""
    prepare_conv_diag!(buffers, propagator, edge, model_data, dlr_grid, dlr_workspace)

Return the convolution operand's per-block diagonals for an edge, keyed by the edge's
hybridization signature. On a cache miss the operand is built into the `conv_dlr` scratch
buffer, its block diagonals are extracted and cached, and the dict is returned. On a hit the
cached diagonals are returned directly — no DLR transform and no dense scatter. Only valid
when the base propagator is diagonal (bare expansion).

Returns a `Dict{Tuple{Int,Int}, Matrix{T}}` mapping each (i,i) sector block to its d×r diagonal.
"""
function prepare_conv_diag!(buffers, propagator, edge, model_data::ModelData,
        dlr_grid::DLRGrid, dlr_workspace)
    sig = conv_signature!(buffers.conv_sig_scratch, get_attached_hybs(edge))
    cached = get(buffers.conv_dlr_cache, sig, nothing)
    if cached === nothing
        conv_dlr = get_conv_dlr(buffers)
        build_conv_dlr!(conv_dlr, get_conv_operand(buffers), propagator, edge,
            model_data, dlr_grid, dlr_workspace)
        cached = cache_conv_diagonals(conv_dlr)
        buffers.conv_dlr_cache[copy(sig)] = cached
    end
    return cached
end

"""
    multiplication_convolution(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from right to left.

# Algorithm

Evaluates diagram backbone from right to left:

```
|Node(M)| --- Edge(M-1) --- |Node(M-1)| --- Edge(M-2) --- ... --- Edge(1) --- |Node(1)|
```

Starting from the rightmost edge, accumulates:

 1. Apply hybridizations attached to edge
 2. Multiply with next node operator
 3. Apply hybridization attached to node
 4. Convolve with next edge's propagator (with its hybridizations)

# Arguments

  - `nodes::Vector{BackBoneNode}`: Diagram nodes (length M)
  - `edges::Vector{BackBoneEdge}`: Diagram edges (length M-1)
  - `dlr_grid::DLRGrid`: DLR grid for τ-space operations
  - `model_data::ModelData`: Model data with operators and hybridization
  - `propagator`: Propagator on DLR τ-grid
  - `buffers`: Pre-allocated work buffers (PropagatorEvalBuffers)

# Returns

Reference to the result buffer containing diagram contribution on DLR τ-grid.

# Notes

Pre-allocated buffers are reused across all diagram evaluations for efficiency.
All operations are in-place to minimize allocations.

# Buffer Usage (5 roles)

  - `result`: Accumulates backbone result
  - `op_product`: Holds operator multiplication output
  - `conv_operand`: Propagator copy for convolution
  - `conv_dlr`: DLR transform of conv operand
  - `result_dlr`: DLR transform of result
"""
function multiplication_convolution(
        nodes::AbstractVector{BackBoneNode}, edges::AbstractVector, dlr_grid::DLRGrid,
        model_data::ModelData{T}, propagator, buffers,
        base_is_diagonal::Bool = true) where {T <: Number}
    length(nodes) == length(edges) + 1 ||
        error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get initial flux
    current_flux = get_flux(propagator)

    # Get result buffer for current flux and copy propagator into it
    result = get_result(buffers, current_flux)
    copyto!(result, propagator)

    # Process edges from right to left
    @inbounds for i in 1:(length(edges) - 1)
        edge = edges[i]
        if i == 1
            for hyb in get_attached_hybs(edge)
                apply_hybridization_inplace!(result, hyb, model_data, is_reverse = false)
            end
        end

        # Multiply with the next node operator
        node = nodes[i + 1]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)

            # Compute new flux after operator multiplication
            new_flux = compute_new_flux(current_flux, op)

            # Get op_product buffer for the new flux
            op_product = get_op_product(buffers, new_flux)
            fill!(op_product, zero(T))

            # Get K-vector for node hybridization (empty if NoHyb)
            K = get_K_vector(node.attached_hyb, model_data, is_reverse = false)

            if !isempty(K)
                # Fused multiplication: op * result -> op_product, with K scaling
                multiply_op_prop_with_K!(op_product, op, result, K)
            else
                # No K-factor, use standard multiplication
                multiply_op_prop!(op_product, op, result)
            end

            # Update current flux and get new result buffer
            current_flux = new_flux
            result = get_result(buffers, current_flux)
            copyto!(result, op_product)
        else
            # No operator at this node, just apply hybridization
            apply_hybridization_inplace!(result, node.attached_hyb, model_data, is_reverse = false)
        end

        # Calculate the convolution with the next edge
        edge_plus1 = edges[i + 1]
        dlr_workspace = model_data.convolution_workspace.dlr_workspace

        result_dlr = get_result_dlr(buffers, current_flux)
        tau2dlr!(result_dlr, dlr_grid, result, dlr_grid.τ, dlr_workspace)

        # Perform convolution - result is written back to result buffer.
        # Forward direction: conv operand is the left factor F.
        if base_is_diagonal
            # F is diagonal within-block: use the cached diagonals and the row-scaling path
            Fdiag_cache = prepare_conv_diag!(buffers, propagator, edge_plus1, model_data,
                dlr_grid, dlr_workspace)
            do_convolution_dlr2tau_Fdiag_propagator!(result, model_data.convolution_workspace,
                dlr_grid, model_data.kernel_cache, Fdiag_cache, result_dlr)
        else
            conv_dlr = get_conv_dlr(buffers)
            build_conv_dlr!(conv_dlr, get_conv_operand(buffers), propagator, edge_plus1,
                model_data, dlr_grid, dlr_workspace)
            do_convolution_dlr2tau!(result, model_data.convolution_workspace, dlr_grid,
                model_data.kernel_cache, conv_dlr, result_dlr)
        end
    end

    return result
end

"""
    multiplication_convolution_reverse(nodes, edges, dlr_grid, model_data, propagator, buffers)

Calculate backbone diagram contribution via multiplication and convolution from left to right.

Similar to `multiplication_convolution` but traverses in reverse direction with swapped K± hybridizations.

# Key Differences from Forward Direction

  - Traverses from left to right (indices M down to 1)
  - Uses `is_reverse=true` for node-attached hybridizations (swaps K+ ↔ K-)
  - Uses `multiply_prop_op!` instead of `multiply_op_prop!` for operator multiplication
  - Convolution operand order is reversed: result first, then conv_operand

# Arguments

Same as `multiplication_convolution`.

# Returns

Reference to the result buffer containing diagram contribution on DLR τ-grid.
"""
function multiplication_convolution_reverse(
        nodes::AbstractVector{BackBoneNode}, edges::AbstractVector, dlr_grid::DLRGrid,
        model_data::ModelData{T}, propagator, buffers,
        base_is_diagonal::Bool = true) where {T <: Number}
    length(nodes) == length(edges) + 1 ||
        error("Number of nodes must be one more than number of edges: got $(length(nodes)) nodes and $(length(edges)) edges")

    # Get initial flux
    current_flux = get_flux(propagator)

    # Get result buffer for current flux and copy propagator into it
    result = get_result(buffers, current_flux)
    copyto!(result, propagator)

    # Process edges from left to right
    @inbounds for i in length(edges):-1:2
        edge = edges[i]
        if i == length(edges)
            for hyb in get_attached_hybs(edge)
                apply_hybridization_inplace!(result, hyb, model_data, is_reverse = false)
            end
        end

        node = nodes[i]
        if node.opt_type !== :empty
            op = get_operator(node, model_data)

            # Compute new flux after operator multiplication (reverse direction)
            new_flux = compute_new_flux_reverse(current_flux, op)

            # Get op_product buffer for the new flux
            op_product = get_op_product(buffers, new_flux)
            fill!(op_product, zero(T))

            # Get K-vector for node hybridization (is_reverse=true swaps K+/K-, empty if NoHyb)
            K = get_K_vector(node.attached_hyb, model_data, is_reverse = true)

            if !isempty(K)
                # Fused multiplication: result * op -> op_product, with K scaling
                multiply_prop_op_with_K!(op_product, result, op, K)
            else
                # No K-factor, use standard multiplication
                multiply_prop_op!(op_product, result, op)
            end

            # Update current flux and get new result buffer
            current_flux = new_flux
            result = get_result(buffers, current_flux)
            copyto!(result, op_product)
        else
            # No operator at this node, just apply hybridization
            apply_hybridization_inplace!(result, node.attached_hyb, model_data, is_reverse = true)
        end

        # Calculate the convolution with the next edge
        edge_minus1 = edges[i - 1]
        dlr_workspace = model_data.convolution_workspace.dlr_workspace

        # Transform to DLR using buffers and workspace (note: order reversed)
        result_dlr = get_result_dlr(buffers, current_flux)
        tau2dlr!(result_dlr, dlr_grid, result, dlr_grid.τ, dlr_workspace)

        # Perform convolution (reversed order) - result is written back to result buffer.
        # Reverse direction: conv operand is the right factor G.
        if base_is_diagonal
            # G is diagonal within-block: use the cached diagonals and the column-scaling path
            Gdiag_cache = prepare_conv_diag!(buffers, propagator, edge_minus1, model_data,
                dlr_grid, dlr_workspace)
            do_convolution_dlr2tau_Gdiag_propagator!(result, model_data.convolution_workspace,
                dlr_grid, model_data.kernel_cache, result_dlr, Gdiag_cache)
        else
            conv_dlr = get_conv_dlr(buffers)
            build_conv_dlr!(conv_dlr, get_conv_operand(buffers), propagator, edge_minus1,
                model_data, dlr_grid, dlr_workspace)
            do_convolution_dlr2tau!(result, model_data.convolution_workspace, dlr_grid,
                model_data.kernel_cache, result_dlr, conv_dlr)
        end
    end

    return result
end

#=============================================================================
# Propagator Backbone Evaluation
=============================================================================#

"""
    eval_propagator_forward_backbone(diagram::MutableBackBone, solver_data, buffers)

Evaluate a single propagator diagram using the unified backbone evaluation.

Called by `eval_propagator_given_order!`.

# Arguments

  - `diagram::MutableBackBone`: The diagram to evaluate
  - `solver_data::SolverData{T}`: Solver data containing model information
  - `buffers`: Pre-allocated work buffers

# Returns

Diagram contribution in imaginary time (Propagator type).
"""
function eval_propagator_forward_backbone(diagram::MutableBackBone, solver_data::SolverData{T}, buffers) where {T <:
                                                                                                                Number}
    bare_propagator = get_bare_propagator(solver_data.model)
    return multiplication_convolution(
        @view(diagram.nodes[1:diagram.M]),
        @view(diagram.edges[1:(diagram.M - 1)]),
        solver_data.model.dlr_grid, solver_data.model,
        bare_propagator.vals,
        buffers
    )
end

#=============================================================================
# Green's Function Backbone Evaluation
=============================================================================#

"""
    eval_green_forward_backbone(diagram::MutableBackBone, split_point, solver_data, propagator, buffers)

Evaluate forward backbone contribution from node 1 to split_point.

# Arguments

  - `diagram::MutableBackBone`: The diagram being evaluated
  - `split_point::Int`: Position of the external annihilation operator
  - `solver_data::SolverData{T}`: Solver data containing model information
  - `propagator`: Propagator to use
  - `buffers`: Evaluation buffers containing forward convolution buffers

# Returns

Forward backbone contribution on DLR τ-grid.
"""
function eval_green_forward_backbone(diagram::MutableBackBone, split_point::Int,
        solver_data::SolverData{T}, propagator, buffers,
        base_is_diagonal::Bool = true) where {T}
    return multiplication_convolution(
        @view(diagram.nodes[1:split_point]),
        @view(diagram.edges[1:(split_point - 1)]),
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator,
        buffers.fwd_buffers,
        base_is_diagonal
    )
end

"""
    eval_green_reverse_backbone(diagram::MutableBackBone, split_point, solver_data, propagator, buffers)

Evaluate reverse backbone contribution from split_point to end.

# Arguments

  - `diagram::MutableBackBone`: The diagram being evaluated
  - `split_point::Int`: Position of the external annihilation operator
  - `solver_data::SolverData{T}`: Solver data containing model information
  - `propagator`: Propagator to use
  - `buffers`: Evaluation buffers containing reverse convolution buffers

# Returns

Reverse backbone contribution on DLR τ-grid.
"""
function eval_green_reverse_backbone(diagram::MutableBackBone, split_point::Int,
        solver_data::SolverData{T}, propagator, buffers,
        base_is_diagonal::Bool = true) where {T}
    return multiplication_convolution_reverse(
        @view(diagram.nodes[split_point:diagram.M]),
        @view(diagram.edges[split_point:(diagram.M - 1)]),
        solver_data.model.dlr_grid,
        solver_data.model,
        propagator,
        buffers.rev_buffers,
        base_is_diagonal
    )
end
