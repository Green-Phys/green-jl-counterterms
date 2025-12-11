# Green's function diagram for both bare and bold expansion
using MPI
using QMBToolkitDLR

"""
    ConvolutionBufferSet{T}

Buffers for a single direction (forward or reverse) of backbone convolution.
Reused across diagrams to avoid allocations during convolution operations.

# Fields
- `backbone_accumulator`: Accumulates the backbone result through multiplication and convolution
- `convolution_operand_buffer`: Prepares the operand for convolution operations
"""
struct ConvolutionBufferSet{T<:Number}
    backbone_accumulator::Array{T,3}
    convolution_operand_buffer::Array{T,3}
end

function ConvolutionBufferSet{T}(n_tau::Int, n_fock::Int) where T<:Number
    return ConvolutionBufferSet{T}(
        zeros(T, n_tau, n_fock, n_fock),
        zeros(T, n_tau, n_fock, n_fock)
    )
end

"""
    GreenConvolutionBuffers{T}

Container for forward and reverse convolution buffer sets.
Both sets are required because forward and reverse convolutions are computed
sequentially but their results (contrib_0_to_tau and contrib_tau_to_beta)
are used simultaneously in operator products.

# Fields
- `fwd`: Forward convolution buffers
- `rev`: Reverse convolution buffers
"""
struct GreenConvolutionBuffers{T<:Number}
    fwd::ConvolutionBufferSet{T}
    rev::ConvolutionBufferSet{T}
end

function GreenConvolutionBuffers{T}(n_tau::Int, n_fock::Int) where T<:Number
    return GreenConvolutionBuffers{T}(
        ConvolutionBufferSet{T}(n_tau, n_fock),
        ConvolutionBufferSet{T}(n_tau, n_fock)
    )
end

"""
    OperatorProductBuffers{T}

Buffers for operator multiplication operations.
Reused across diagrams to avoid allocations during matrix products.

# Fields
- `operator_product_buffer`: General-purpose buffer for operator multiplications (shared across convolutions)
- `annihilation_product_buffer`: Holds products involving annihilation operators
- `final_product_buffer`: Holds final products before accumulation
- `trace_result_buffer`: Holds trace computation results
"""
struct OperatorProductBuffers{T<:Number}
    operator_product_buffer::Array{T,3}
    annihilation_product_buffer::Array{T,3}
    final_product_buffer::Array{T,3}
    trace_result_buffer::Vector{Float64}  # Trace is always real
end

function OperatorProductBuffers{T}(n_tau::Int, n_fock::Int) where T<:Number
    return OperatorProductBuffers{T}(
        zeros(T, n_tau, n_fock, n_fock),
        zeros(T, n_tau, n_fock, n_fock),
        zeros(T, n_tau, n_fock, n_fock),
        zeros(Float64, n_tau)
    )
end

function gen_green_given_order(expansion_order::Int, n_flavors::Int, exponents::Vector{Float64}, is_bold::Bool=false)

    if is_bold
        if expansion_order == 1
            diagrams1 = gen_green_given_topology(3, 2, [], 1.0, n_flavors, exponents)
            diagrams = stack([diagrams1], dims=1)
        elseif expansion_order == 2
            diagrams1 = gen_green_given_topology(5, 3, [(4, 2)], -1.0, n_flavors, exponents)
            diagrams = stack([diagrams1], dims=1)
        elseif expansion_order == 3
            diagrams1 = gen_green_given_topology(7, 3, [(6, 4), (5, 2)], 1.0, n_flavors, exponents)
            diagrams2 = gen_green_given_topology(7, 4, [(6, 3), (5, 2)], -1.0, n_flavors, exponents)
            diagrams3 = gen_green_given_topology(7, 5, [(6, 3), (4, 2)], 1.0, n_flavors, exponents)
            diagrams4 = gen_green_given_topology(7, 4, [(6, 2), (5, 3)], 1.0, n_flavors, exponents)
            diagrams = stack([diagrams1, diagrams2, diagrams3, diagrams4], dims=1)
        else
            error("Unsupported expansion order for Green's function: $expansion_order")

        end
    else
        pairings_with_crossings = _generate_pairings_with_hole(expansion_order)
        diagrams_list = Vector{Array{BackBone,3}}()
        for (pairing, annihilation_pos, crossing_count) in pairings_with_crossings
            crossing_parity = (-1.0)^crossing_count
            new_diagrams = gen_green_given_topology(2 * expansion_order + 1, annihilation_pos, pairing, crossing_parity, n_flavors, exponents)
            push!(diagrams_list, new_diagrams)
        end
        diagrams = stack(diagrams_list, dims=1)
    end
    return diagrams
end


function gen_green_given_topology(M::Int, annihilation_pos::Int, connected_pairs::Vector{Tuple{Int,Int}}, crossing_parity::Float64, n_flavors::Int, exponents::Vector{Float64})
    external_opts_loc = [1, annihilation_pos]
    empty_node = BackBoneNode(:empty, -1, -1, NoHyb())
    p = length(exponents)

    # Validate connected_pairs structure
    @assert length(connected_pairs) == (M - 3) ÷ 2 "Number of connected pairs does not match M"
    @assert all(x -> x[1] > x[2], connected_pairs) "Connected pairs should have first index larger than second index"
    @assert issorted([x[1] for x in connected_pairs]; rev=true) "Connected pairs are not ordered properly"
    @assert sort(vcat([x[1] for x in connected_pairs], [x[2] for x in connected_pairs])) == setdiff(collect(2:M-1), [annihilation_pos]) "Connected pairs do not cover all internal nodes"

    base_nodes = Vector{BackBoneNode}(undef, M)
    base_nodes[1] = empty_node
    base_nodes[annihilation_pos] = empty_node
    base_nodes[M] = empty_node
    base_sign = -1 * crossing_parity #-1 comes from the definition of Green's function

    return gen_diagram_combinations(:green_diagram, M, connected_pairs, n_flavors, exponents, base_nodes, base_sign, external_opts_loc)
end


"""
    eval_green_given_order!(solver_data, diagrams, is_bold, comm)

Evaluate all Green's function diagrams at a given expansion order and accumulate results.

# Arguments
- `solver_data::SolverData`: The solver data containing model information and calculation results
- `diagrams::Array{BackBone,4}`: 4D array of diagrams organized by (topology, direction, flavor, exponent)
- `is_bold::Bool`: Whether to use bold expansion (true) or bare expansion (false)
- `comm::MPI.Comm`: MPI communicator for parallel execution

Updates `solver_data.calc.green.vals` by accumulating contributions from all diagrams.
Uses MPI parallelization to distribute diagram evaluation across ranks.
"""
function eval_green_given_order!(solver_data::SolverData{T}, diagrams::Array{BackBone,4}, is_bold::Bool, comm::MPI.Comm) where {T<:Number}
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    n_topology = size(diagrams, 1)
    n_directions = size(diagrams, 2)
    n_line_flavors = size(diagrams, 3)
    n_hyb_exponents = size(diagrams, 4)

    if is_bold
        propagator = solver_data.calc.propagator
    else
        propagator = solver_data.model.bare_propagator
    end

    # Create local accumulator for this rank's work (only for this order)
    local_result = zeros(T, size(solver_data.calc.green.vals))

    # ============================================================================
    # Buffer Allocation Strategy
    # ============================================================================
    # Pre-allocate work buffers that are reused across all diagram evaluations
    # to avoid repeated allocations in the hot loop. Buffers are grouped into
    # structs for organizational clarity.
    #
    # Buffer reuse pattern for Green's function evaluation:
    #
    # 1. OperatorProductBuffers (op_buffers):
    #    - operator_product_buffer: Shared between fwd/rev convolutions for hybridization and operator products
    #    - annihilation_product_buffer: Temporary storage for annihilation operator multiplication
    #    - final_product_buffer: Holds final operator products before accumulation
    #    - trace_result_buffer: Holds trace results before adding to local_result
    #
    # 2. GreenConvolutionBuffers (conv_buffers):
    #    - Contains separate fwd and rev buffer sets (ConvolutionBufferSet)
    #    - Each set has: backbone_accumulator and convolution_operand_buffer
    #    - Separate fwd/rev sets required because contrib_0_to_tau and contrib_tau_to_beta
    #      are computed sequentially but both results are used together in operator products
    #
    # Each diagram evaluation reuses the same buffers, overwriting previous values.
    # This is safe because contributions are accumulated into backbone_0_to_beta_with_ann
    # and local_result after each use.
    # ============================================================================
    n_tau = length(solver_data.model.dlr_grid.tau)
    n_fock = solver_data.model.n_fock

    op_buffers = OperatorProductBuffers{T}(n_tau, n_fock)
    conv_buffers = GreenConvolutionBuffers{T}(n_tau, n_fock)

    # Distribute topology loop across ranks: rank processes indices rank+1:nprocs:n_topology
    my_topologies = rank+1:nprocs:n_topology

    for i in my_topologies
        for j in 1:n_directions
            external_opts_loc = diagrams[i, j, 1, 1].external_opts_loc
            @assert length(external_opts_loc) == 2 && external_opts_loc[1] == 1 "External operator locations are invalid."
            backbone_0_to_beta_with_ann = [zeros(T, size(propagator.vals)) for i in 1:solver_data.model.n_flavors_physical]
            for k in 1:n_line_flavors
                for l in 1:n_hyb_exponents
                    diagram = diagrams[i, j, k, l]
                    @assert diagram.external_opts_loc == external_opts_loc "External operator locations do not match among diagrams of the same topology and direction."
                    contrib_0_to_tau = multiplication_convolution(diagram.nodes[1:external_opts_loc[2]], diagram.edges[1:external_opts_loc[2]-1], solver_data.model.dlr_grid, solver_data.model, propagator, op_buffers.operator_product_buffer, conv_buffers.fwd.backbone_accumulator, conv_buffers.fwd.convolution_operand_buffer)
                    contrib_tau_to_beta = multiplication_convolution_reverse(diagram.nodes[external_opts_loc[2]:end], diagram.edges[external_opts_loc[2]:end], solver_data.model.dlr_grid, solver_data.model, propagator, op_buffers.operator_product_buffer, conv_buffers.rev.backbone_accumulator, conv_buffers.rev.convolution_operand_buffer)
                    contrib_tau_to_beta = do_reflection_tau2tau(solver_data.model.dlr_grid, contrib_tau_to_beta)
                    prefactor = compute_diagram_prefactor(diagram, solver_data.model.K0_inverse)
                    for ann_index in 1:solver_data.model.n_flavors_physical
                        # Use in-place versions: annihilation_product_buffer = ann_opt * contrib_0_to_tau, then final_product_buffer = contrib_tau_to_beta * annihilation_product_buffer
                        _multiplication_23!(op_buffers.annihilation_product_buffer, solver_data.model.ann_opts[ann_index], contrib_0_to_tau)
                        _multiplication_33!(op_buffers.final_product_buffer, contrib_tau_to_beta, op_buffers.annihilation_product_buffer)
                        backbone_0_to_beta_with_ann[ann_index] .+= prefactor .* op_buffers.final_product_buffer
                    end
                end
            end
            for ann_index in 1:solver_data.model.n_flavors_physical
                for cre_index in 1:solver_data.model.n_flavors_physical
                    # Use in-place version: annihilation_product_buffer = backbone * cre_opt, then trace
                    _multiplication_32!(op_buffers.annihilation_product_buffer, backbone_0_to_beta_with_ann[ann_index], solver_data.model.cre_opts[cre_index])
                    _trace_3!(op_buffers.trace_result_buffer, op_buffers.annihilation_product_buffer)
                    local_result[:, ann_index, cre_index] .+= op_buffers.trace_result_buffer
                end
            end
        end
    end

    # Sum results from all ranks for this order, then ADD to existing green values
    order_result = MPI.Allreduce(local_result, MPI.SUM, comm)
    solver_data.calc.green.vals .+= order_result
end


function _generate_pairings_with_hole(N::Int)
    # Points: 2, 3, ..., 2N
    all_points = collect(2:(2*N))

    final_output = []

    for annihilation_pos in all_points
        # Remove the hole from the pool
        current_pool = filter(x -> x != annihilation_pos, all_points)

        pairings = recurse_pairs(current_pool)

        # External Contribution: Count points smaller than annihilation_pos
        # Since points are 2, 3... this is simply (annihilation_pos - 2)
        # But we use a generic count for robustness.
        external_crossings = count(p -> p < annihilation_pos, all_points)

        for pairs in pairings
            # Internal crossings (arch vs arch) using shared function
            internal_crossings = count_pair_crossings(pairs)
            total_crossings = internal_crossings + external_crossings

            push!(final_output, (pairs, annihilation_pos, total_crossings))
        end
    end

    return final_output
end