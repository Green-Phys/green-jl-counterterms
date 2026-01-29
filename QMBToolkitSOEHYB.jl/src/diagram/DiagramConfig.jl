#=
DiagramConfig.jl - Configuration caching for diagram generation

This module provides infrastructure for memory-efficient diagram generation by:
1. Caching all configuration combinations (topology, direction, flavor, exponent)
2. Enabling index-based diagram generation without storing all diagrams
3. Supporting single diagram generation from cached configs

Key types:
- TopologyConfig: Stores pairing structure and crossing information
- DiagramConfigCache: Complete cache for an expansion order

Key functions:
- build_config_cache: Build configuration cache for a given order
- total_diagrams: Get total diagram count from cache
- linear_to_config_indices: Map linear index to 4D config indices
- generate_diagram_from_configs: Generate single diagram from configs
- generate_diagram_at_index: Generate single diagram at linear index
=#

#=============================================================================
# Configuration Structures
=============================================================================#

"""
    TopologyConfig

Configuration for a single diagram topology.

# Fields

  - `pairing::Vector{Tuple{Int,Int}}`: Connected pairs of nodes (hybridization lines)
  - `crossing_parity::Float64`: Sign factor from crossing count, (-1)^crossings
  - `annihilation_pos::Int`: Position of external annihilation operator (Green only, -1 for propagator)
  - `M::Int`: Total number of nodes in the backbone
"""
struct TopologyConfig
    pairing::Vector{Tuple{Int, Int}}
    crossing_parity::Float64
    annihilation_pos::Int
    M::Int
end

"""
    DiagramConfigCache

Complete configuration cache for diagram generation at a specific expansion order.

Stores all possible combinations of:

  - Topologies (pairing structures)
  - Hybridization directions (forward/backward for each line)
  - Flavor assignments (orbital index for each line)
  - Exponent indices (SOE index for each line)

# Fields

  - `order::Int`: Expansion order
  - `diagram_type::Symbol`: :propagator_diagram or :green_diagram
  - `n_hyb_lines::Int`: Number of hybridization lines (L)
  - `n_topologies::Int`: Number of distinct topologies
  - `n_directions::Int`: Number of direction combinations (2^L)
  - `n_flavors_combo::Int`: Number of flavor combinations (n_flavors^L)
  - `n_exponents_combo::Int`: Number of exponent combinations (p^L)
  - `topologies::Vector{TopologyConfig}`: All topology configurations
  - `directions::Vector{NTuple}`: All direction combinations
  - `flavors::Vector{NTuple}`: All flavor index combinations
  - `exponent_indices::Vector{NTuple}`: All exponent index combinations
  - `exponents::Vector{Float64}`: Actual exponent values (for diagram generation)
  - `n_flavors::Int`: Number of flavors (for diagram generation)
"""
struct DiagramConfigCache{L}
    order::Int
    diagram_type::Symbol
    n_hyb_lines::Int
    n_topologies::Int
    n_directions::Int
    n_flavors_combo::Int
    n_exponents_combo::Int
    topologies::Vector{TopologyConfig}
    directions::Vector{NTuple{L, Symbol}}
    flavors::Vector{NTuple{L, Int}}
    exponent_indices::Vector{NTuple{L, Int}}
    exponents::Vector{Float64}
    n_flavors::Int
end

#=============================================================================
# Helper Functions for Generating Combinations
=============================================================================#

"""
    _generate_direction_combinations(L::Int)

Generate all 2^L combinations of hybridization directions.
Each line can be :forward or :backward.
"""
function _generate_direction_combinations(L::Int)
    if L == 0
        return [()]
    end
    directions = Vector{NTuple{L, Symbol}}()
    for combo in Iterators.product(fill((:forward, :backward), L)...)
        push!(directions, combo)
    end
    return directions
end

"""
    _generate_index_combinations(L::Int, n::Int)

Generate all n^L combinations of indices from 1 to n.
Used for both flavor and exponent index combinations.
"""
function _generate_index_combinations(L::Int, n::Int)
    if L == 0
        return [()]
    end
    combinations = Vector{NTuple{L, Int}}()
    for combo in Iterators.product(fill(1:n, L)...)
        push!(combinations, combo)
    end
    return combinations
end

#=============================================================================
# Topology Generation (without creating diagrams)
=============================================================================#

"""
    _build_propagator_topologies(order::Int)

Build topology configurations for propagator diagrams at given order.
Uses the same pairing logic as gen_propagator.jl but only stores the configuration.
"""
function _build_propagator_topologies(order::Int)
    if order == 0
        # Order 0: single topology with no pairings, M=2 nodes
        return [TopologyConfig(Tuple{Int, Int}[], 1.0, -1, 2)]
    end

    pairings_with_crossings = _generate_pairings(order)
    M = 2 + 2 * order

    topologies = Vector{TopologyConfig}()
    for (pairing, crossing_count) in pairings_with_crossings
        crossing_parity = (-1.0)^crossing_count
        push!(topologies, TopologyConfig(pairing, crossing_parity, -1, M))
    end
    return topologies
end

"""
    _build_green_topologies(order::Int, is_bold::Bool)

Build topology configurations for Green's function diagrams at given order.
Uses the same pairing logic as gen_green.jl but only stores the configuration.
"""
function _build_green_topologies(order::Int, is_bold::Bool)
    if is_bold
        # Bold expansion has hardcoded topologies
        if order == 1
            M = 3
            return [TopologyConfig(Tuple{Int, Int}[], 1.0, 2, M)]
        elseif order == 2
            M = 5
            return [TopologyConfig([(4, 2)], -1.0, 3, M)]
        elseif order == 3
            M = 7
            return [
                TopologyConfig([(6, 4), (5, 2)], 1.0, 3, M),
                TopologyConfig([(6, 3), (5, 2)], -1.0, 4, M),
                TopologyConfig([(6, 3), (4, 2)], 1.0, 5, M),
                TopologyConfig([(6, 2), (5, 3)], 1.0, 4, M)
            ]
        else
            error("Unsupported expansion order for bold Green's function: $order")
        end
    else
        # Bare expansion uses generated pairings with hole
        pairings_with_crossings = _generate_pairings_with_hole(order)
        M = 2 * order + 1

        topologies = Vector{TopologyConfig}()
        for (pairing, annihilation_pos, crossing_count) in pairings_with_crossings
            crossing_parity = (-1.0)^crossing_count
            push!(topologies, TopologyConfig(pairing, crossing_parity, annihilation_pos, M))
        end
        return topologies
    end
end

#=============================================================================
# Main Cache Building Functions
=============================================================================#

"""
    build_config_cache(order::Int, n_flavors::Int, exponents::Vector{Float64},
                       diagram_type::Symbol; is_bold::Bool=false)

Build a complete configuration cache for diagram generation.

# Arguments

  - `order::Int`: Expansion order
  - `n_flavors::Int`: Number of fermion flavors
  - `exponents::Vector{Float64}`: SOE exponents
  - `diagram_type::Symbol`: :propagator_diagram or :green_diagram
  - `is_bold::Bool`: Whether using bold expansion (only for green_diagram)

# Returns

  - `DiagramConfigCache`: Complete cache with all configurations
"""
function build_config_cache(order::Int, n_flavors::Int, exponents::Vector{Float64},
        diagram_type::Symbol; is_bold::Bool = false)
    p = length(exponents)

    # Determine number of hybridization lines
    if diagram_type == :propagator_diagram
        L = order
    elseif diagram_type == :green_diagram
        L = order - 1
    else
        error("Unsupported diagram type: $diagram_type")
    end

    # Build topologies
    if diagram_type == :propagator_diagram
        topologies = _build_propagator_topologies(order)
    else
        topologies = _build_green_topologies(order, is_bold)
    end

    # Build all combinations
    directions = _generate_direction_combinations(L)
    flavors = _generate_index_combinations(L, n_flavors)
    exponent_indices = _generate_index_combinations(L, p)

    n_topologies = length(topologies)
    n_directions = length(directions)
    n_flavors_combo = length(flavors)
    n_exponents_combo = length(exponent_indices)

    return DiagramConfigCache{L}(
        order,
        diagram_type,
        L,
        n_topologies,
        n_directions,
        n_flavors_combo,
        n_exponents_combo,
        topologies,
        directions,
        flavors,
        exponent_indices,
        exponents,
        n_flavors
    )
end

"""
    total_diagrams(cache::DiagramConfigCache)

Get the total number of diagrams represented by this cache.
"""
function total_diagrams(cache::DiagramConfigCache)
    return cache.n_topologies * cache.n_directions * cache.n_flavors_combo * cache.n_exponents_combo
end

"""
    config_dimensions(cache::DiagramConfigCache)

Get the 4D dimensions of the diagram array: (n_topo, n_dir, n_flav, n_exp).
This matches the shape returned by the original gen_*_given_order functions.
"""
function config_dimensions(cache::DiagramConfigCache)
    return (cache.n_topologies, cache.n_directions, cache.n_flavors_combo, cache.n_exponents_combo)
end

#=============================================================================
# Index Mapping Functions
=============================================================================#

"""
    linear_to_config_indices(cache::DiagramConfigCache, linear_idx::Int)

Convert a linear index (1-based) to 4D config indices (topo_idx, dir_idx, flav_idx, exp_idx).

The linear index corresponds to a flattened 4D array with dimensions:
(n_topologies, n_directions, n_flavors_combo, n_exponents_combo)

Julia uses column-major order, so the FIRST index varies fastest.
Linear index `i` maps to:

  - topo_idx = (i - 1) % T + 1
  - dir_idx  = ((i - 1) ÷ T) % D + 1
  - flav_idx = ((i - 1) ÷ (T * D)) % F + 1
  - exp_idx  = ((i - 1) ÷ (T * D * F)) % E + 1

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `linear_idx::Int`: Linear index (1-based, from 1 to total_diagrams)

# Returns

  - `Tuple{Int,Int,Int,Int}`: (topo_idx, dir_idx, flav_idx, exp_idx), all 1-based
"""
function linear_to_config_indices(cache::DiagramConfigCache, linear_idx::Int)
    @boundscheck @assert 1 <= linear_idx <= total_diagrams(cache) "Linear index $linear_idx out of range [1, $(total_diagrams(cache))]"

    T = cache.n_topologies
    D = cache.n_directions
    F = cache.n_flavors_combo
    E = cache.n_exponents_combo

    idx = linear_idx - 1  # Convert to 0-based for arithmetic

    topo_idx = idx % T + 1
    idx = idx ÷ T

    dir_idx = idx % D + 1
    idx = idx ÷ D

    flav_idx = idx % F + 1
    idx = idx ÷ F

    exp_idx = idx % E + 1

    return (topo_idx, dir_idx, flav_idx, exp_idx)
end

"""
    get_configs_at_index(cache::DiagramConfigCache, linear_idx::Int)

Get the configuration tuples for a given linear index.

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `linear_idx::Int`: Linear index (1-based)

# Returns

  - `Tuple`: (topology::TopologyConfig, direction::NTuple, flavor::NTuple, exponent_indices::NTuple)
"""
function get_configs_at_index(cache::DiagramConfigCache, linear_idx::Int)
    topo_idx, dir_idx, flav_idx, exp_idx = linear_to_config_indices(cache, linear_idx)

    return (
        cache.topologies[topo_idx],
        cache.directions[dir_idx],
        cache.flavors[flav_idx],
        cache.exponent_indices[exp_idx]
    )
end

#=============================================================================
# Single Diagram Generation
=============================================================================#

"""
    generate_diagram_from_configs(cache::DiagramConfigCache, topo_config::TopologyConfig,
                                   directions::NTuple, flavors::NTuple, exponent_indices::NTuple)

Generate a single BackBone diagram from pre-cached configuration tuples.

This function creates a single diagram without storing all diagrams in memory,
enabling memory-efficient batched generation.

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache (provides exponents, n_flavors, diagram_type)
  - `topo_config::TopologyConfig`: Topology configuration (pairing, crossing_parity, annihilation_pos, M)
  - `directions::NTuple`: Direction for each hybridization line (:forward or :backward)
  - `flavors::NTuple`: Flavor index for each hybridization line
  - `exponent_indices::NTuple`: Exponent index for each hybridization line

# Returns

  - `BackBone`: The generated diagram
"""
function generate_diagram_from_configs(
        cache::DiagramConfigCache{L},
        topo_config::TopologyConfig,
        directions::NTuple{L, Symbol},
        flavors::NTuple{L, Int},
        exponent_indices::NTuple{L, Int}
) where {L}
    M = topo_config.M
    p = length(cache.exponents)

    # Set up base_nodes based on diagram type
    empty_node = BackBoneNode(:empty, -1, -1, NoHyb())
    base_nodes = Vector{BackBoneNode}(undef, M)

    if cache.diagram_type == :propagator_diagram
        # Propagator: only boundary nodes are empty
        base_nodes[1] = empty_node
        base_nodes[M] = empty_node
        external_opts_loc = Int[]
        base_sign = topo_config.crossing_parity
    elseif cache.diagram_type == :green_diagram
        # Green: boundary nodes and annihilation_pos are empty
        base_nodes[1] = empty_node
        base_nodes[topo_config.annihilation_pos] = empty_node
        base_nodes[M] = empty_node
        external_opts_loc = [1, topo_config.annihilation_pos]
        base_sign = -1.0 * topo_config.crossing_parity  # -1 from Green's function definition
    else
        error("Unsupported diagram type: $(cache.diagram_type)")
    end

    # Initialize working state
    nodes = copy(base_nodes)
    edges = [BackBoneEdge(HybridizationState[]) for _ in 1:(M - 1)]
    prefactor_Kexps = zeros(Int, p)
    sign = base_sign

    # Process each connected pair
    connected_pairs = topo_config.pairing
    for (idx, (i, j)) in enumerate(connected_pairs)
        sign = process_connected_pair!(
            i, j,
            directions[idx],
            flavors[idx],
            exponent_indices[idx],
            cache.exponents,
            nodes,
            edges,
            prefactor_Kexps,
            sign
        )
    end

    return BackBone(cache.diagram_type, sign, prefactor_Kexps, nodes, edges, external_opts_loc)
end

"""
    generate_diagram_at_index(cache::DiagramConfigCache, linear_idx::Int)

Generate a single diagram at the given linear index.

Convenience function that combines index mapping and diagram generation.

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `linear_idx::Int`: Linear index (1-based)

# Returns

  - `BackBone`: The generated diagram
"""
function generate_diagram_at_index(cache::DiagramConfigCache, linear_idx::Int)
    topo_config, directions, flavors, exponent_indices = get_configs_at_index(cache, linear_idx)
    return generate_diagram_from_configs(cache, topo_config, directions, flavors, exponent_indices)
end

#=============================================================================
# Zero-Allocation Diagram Generation (MutableBackBone)
=============================================================================#

"""
    fill_backbone_from_configs!(bb::MutableBackBone, cache::DiagramConfigCache,
                                topo_config::TopologyConfig, directions::NTuple,
                                flavors::NTuple, exponent_indices::NTuple)

Fill a pre-allocated MutableBackBone with diagram configuration.
The backbone must be reset before calling this function.

# Arguments

  - `bb::MutableBackBone`: Pre-allocated and reset backbone to fill
  - `cache::DiagramConfigCache`: The configuration cache
  - `topo_config::TopologyConfig`: Topology configuration
  - `directions::NTuple`: Direction for each hybridization line
  - `flavors::NTuple`: Flavor index for each hybridization line
  - `exponent_indices::NTuple`: Exponent index for each hybridization line

# Returns

  - `MutableBackBone`: The filled backbone (same object as input)
"""
function fill_backbone_from_configs!(
        bb::MutableBackBone,
        cache::DiagramConfigCache{L},
        topo_config::TopologyConfig,
        directions::NTuple{L, Symbol},
        flavors::NTuple{L, Int},
        exponent_indices::NTuple{L, Int}
) where {L}
    M = topo_config.M
    p = length(cache.exponents)
    bb.M = M
    bb.diagram_type = cache.diagram_type

    # Set up empty node template
    empty_node = BackBoneNode(:empty, -1, -1, NoHyb())

    # Initialize nodes based on diagram type
    if cache.diagram_type == :propagator_diagram
        bb.nodes[1] = empty_node
        bb.nodes[M] = empty_node
        bb.sign = topo_config.crossing_parity
    elseif cache.diagram_type == :green_diagram
        bb.nodes[1] = empty_node
        bb.nodes[topo_config.annihilation_pos] = empty_node
        bb.nodes[M] = empty_node
        bb.external_opts_loc[1] = 1
        bb.external_opts_loc[2] = topo_config.annihilation_pos
        bb.sign = -1.0 * topo_config.crossing_parity
    else
        error("Unsupported diagram type: $(cache.diagram_type)")
    end

    # Process each connected pair
    connected_pairs = topo_config.pairing
    for (idx, (i, j)) in enumerate(connected_pairs)
        bb.sign = process_connected_pair_mutable!(
            i, j,
            directions[idx],
            flavors[idx],
            exponent_indices[idx],
            cache.exponents,
            bb.nodes,
            bb.edges,
            bb.prefactor_Kexps,
            bb.sign
        )
    end

    return bb
end

"""
    allocate_mutable_backbone(cache::DiagramConfigCache)

Allocate a MutableBackBone sized for diagrams from this cache.

The backbone is sized for the maximum M across all topologies, with sufficient
edge capacity for the worst-case number of hybridizations per edge.
"""
function allocate_mutable_backbone(cache::DiagramConfigCache)
    # Find maximum M across all topologies
    max_M = maximum(t.M for t in cache.topologies)

    # Maximum hybridizations per edge is L (all lines could attach to same edge)
    max_hybs_per_edge = cache.n_hyb_lines

    # Number of exponents
    p = length(cache.exponents)

    return MutableBackBone(max_M, p, max_hybs_per_edge)
end

#=============================================================================
# Flux Pre-Filtering Infrastructure
=============================================================================#

using QMBToolkitED: Basis, QuantumNumber

"""
    DiagramGroup

A group of diagrams sharing the same (topology, direction, flavor) configuration.
All diagrams in a group have the same flux sequence and operator structure,
differing only in exponent indices.

# Fields

  - `topo_idx::Int`: Index into cache.topologies
  - `dir_idx::Int`: Index into cache.directions
  - `flav_idx::Int`: Index into cache.flavors
  - `n_exponent_combos::Int`: Number of exponent combinations (diagrams in this group)
"""
struct DiagramGroup
    topo_idx::Int
    dir_idx::Int
    flav_idx::Int
    n_exponent_combos::Int
end

"""
    check_flux_sequence(topo_config, directions, flavors, basis, diagram_type, valid_fluxes) -> Bool

Check if the operator sequence produces only valid intermediate fluxes.

Returns true if all intermediate fluxes encountered during backbone evaluation
would have non-empty block structure in the given basis.

For propagator diagrams: single flux evolution through the entire backbone.
For Green's function diagrams: separate forward and reverse backbone evolutions,
plus validation that external operators can produce valid combined fluxes.

# Arguments

  - `topo_config::TopologyConfig`: Topology configuration
  - `directions::NTuple{L,Symbol}`: Direction for each hybridization line
  - `flavors::NTuple{L,Int}`: Flavor index for each hybridization line
  - `basis::Basis`: The basis defining valid fluxes
  - `diagram_type::Symbol`: :propagator_diagram or :green_diagram
  - `valid_fluxes::Set{QN}`: Pre-computed set of valid flux values

# Returns

  - `Bool`: true if all intermediate fluxes are valid, false otherwise
"""
function check_flux_sequence(
        topo_config::TopologyConfig,
        directions::NTuple{L, Symbol},
        flavors::NTuple{L, Int},
        basis::Basis,
        diagram_type::Symbol,
        valid_fluxes::Set{QN}
) where {L, QN <: QuantumNumber}
    M = topo_config.M

    # Build operator sequence with their fluxes
    # Each entry: (node_position, operator_flux)
    node_fluxes = Vector{Tuple{Int, QN}}()

    for (idx, (i, j)) in enumerate(topo_config.pairing)
        flavor = flavors[idx]
        flavor_qn = basis.flavor_qns[flavor]

        if directions[idx] == :forward
            # cdag at node i, c at node j
            push!(node_fluxes, (i, flavor_qn))      # cdag adds +flavor_qn
            push!(node_fluxes, (j, -flavor_qn))     # c adds -flavor_qn
        else  # :backward
            # c at node i, cdag at node j
            push!(node_fluxes, (i, -flavor_qn))     # c adds -flavor_qn
            push!(node_fluxes, (j, flavor_qn))      # cdag adds +flavor_qn
        end
    end

    # Sort by node position
    sort!(node_fluxes, by = x -> x[1])

    if diagram_type == :propagator_diagram
        # Propagator: single flux evolution through entire backbone
        # Nodes 1 and M are empty (boundary)
        current_flux = zero(QN)

        for (node_pos, op_flux) in node_fluxes
            if node_pos == 1 || node_pos == M
                continue
            end

            current_flux = current_flux + op_flux

            if current_flux ∉ valid_fluxes
                return false
            end
        end

        return true
    else  # :green_diagram
        # Green's function: SEPARATE forward and reverse backbone evolutions
        # Forward: nodes 1 to split_point (1 and split_point are empty)
        # Reverse: nodes split_point to M (split_point and M are empty)
        # Each starts from zero flux independently!

        split_point = topo_config.annihilation_pos

        # Check forward backbone flux evolution (nodes 2 to split_point-1)
        fwd_flux = zero(QN)
        for (node_pos, op_flux) in node_fluxes
            if node_pos <= 1 || node_pos >= split_point
                continue  # Not in forward backbone
            end

            fwd_flux = fwd_flux + op_flux

            if fwd_flux ∉ valid_fluxes
                return false
            end
        end

        # Check reverse backbone flux evolution (nodes split_point+1 to M-1)
        # IMPORTANT: reverse backbone processes nodes in DECREASING order (M-1 down to split_point+1)
        # We need to check intermediate fluxes in the same order
        rev_node_fluxes = [(pos, flux) for (pos, flux) in node_fluxes
                           if pos > split_point && pos < M]
        sort!(rev_node_fluxes, by = x -> -x[1])  # Sort in decreasing order

        rev_flux = zero(QN)
        for (node_pos, op_flux) in rev_node_fluxes
            rev_flux = rev_flux + op_flux

            if rev_flux ∉ valid_fluxes
                return false
            end
        end

        # Check that there exists at least one (α, β) combination where:
        # 1. ann_product_flux = fwd_flux + (-flavor_qn[α]) is valid
        # 2. final_flux = rev_flux + ann_product_flux is valid (for multiplication)
        # 3. total_flux = final_flux + flavor_qn[β] = 0 (for trace)
        #
        # Condition 3 means: rev_flux + fwd_flux - qn[α] + qn[β] = 0
        # So: qn[β] - qn[α] = -(fwd_flux + rev_flux)
        #
        # We need to check if there exists valid (α, β) pair

        n_flavors = length(basis.flavor_qns)
        has_valid_external = false

        for α in 1:n_flavors
            α_qn = basis.flavor_qns[α]
            ann_product_flux = fwd_flux + (-α_qn)

            # Check if ann_product_flux is valid
            if ann_product_flux ∉ valid_fluxes
                continue
            end

            for β in 1:n_flavors
                β_qn = basis.flavor_qns[β]
                # After multiplying with reverse backbone: rev_flux + ann_product_flux
                # After applying c†_β: rev_flux + ann_product_flux + β_qn
                # For non-zero trace, this must be zero
                total_flux = rev_flux + ann_product_flux + β_qn

                if total_flux == zero(QN)
                    has_valid_external = true
                    break
                end
            end

            if has_valid_external
                break
            end
        end

        return has_valid_external
    end
end

"""
    build_valid_groups(cache::DiagramConfigCache, basis::Basis) -> Vector{DiagramGroup}

Build list of valid diagram groups by pre-filtering invalid flux sequences.

A group is valid if its (topology, direction, flavor) combination produces
only valid intermediate fluxes during backbone evaluation.

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `basis::Basis`: The basis defining valid fluxes

# Returns

  - `Vector{DiagramGroup}`: List of valid groups, each containing n_exponent_combos diagrams
"""
function build_valid_groups(cache::DiagramConfigCache{L}, basis::Basis) where {L}
    # Pre-compute valid fluxes once (avoid rebuilding for each group check)
    QN = typeof(basis.sectors[1].qn)
    valid_fluxes = Set{QN}()
    for src_sector in basis.sectors
        for tgt_sector in basis.sectors
            flux = tgt_sector.qn + (-src_sector.qn)
            push!(valid_fluxes, flux)
        end
    end

    groups = DiagramGroup[]

    for topo_idx in 1:cache.n_topologies
        topo_config = cache.topologies[topo_idx]

        for dir_idx in 1:cache.n_directions
            directions = cache.directions[dir_idx]

            for flav_idx in 1:cache.n_flavors_combo
                flavors = cache.flavors[flav_idx]

                # Check if this combination has valid flux sequence
                if check_flux_sequence(topo_config, directions, flavors, basis,
                    cache.diagram_type, valid_fluxes)
                    push!(groups, DiagramGroup(
                        topo_idx, dir_idx, flav_idx, cache.n_exponents_combo
                    ))
                end
            end
        end
    end

    return groups
end

"""
    total_diagrams_in_groups(groups::Vector{DiagramGroup}) -> Int

Get total number of diagrams across all valid groups.
"""
function total_diagrams_in_groups(groups::Vector{DiagramGroup})
    return sum(g.n_exponent_combos for g in groups; init = 0)
end

"""
    generate_diagram_from_group(cache::DiagramConfigCache, group::DiagramGroup, exp_idx::Int)

Generate a diagram from a group and exponent index.

# Arguments

  - `cache::DiagramConfigCache`: The configuration cache
  - `group::DiagramGroup`: The diagram group
  - `exp_idx::Int`: Index into cache.exponent_indices (1-based)

# Returns

  - `BackBone`: The generated diagram
"""
function generate_diagram_from_group(
        cache::DiagramConfigCache{L}, group::DiagramGroup, exp_idx::Int) where {L}
    return generate_diagram_from_configs(
        cache,
        cache.topologies[group.topo_idx],
        cache.directions[group.dir_idx],
        cache.flavors[group.flav_idx],
        cache.exponent_indices[exp_idx]
    )
end

"""
    fill_backbone_from_group!(bb::MutableBackBone, cache::DiagramConfigCache,
                              group::DiagramGroup, exp_idx::Int)

Fill a MutableBackBone from a group and exponent index (zero-allocation version).
The backbone must be reset before calling this function.

# Arguments

  - `bb::MutableBackBone`: Pre-allocated and reset backbone to fill
  - `cache::DiagramConfigCache`: The configuration cache
  - `group::DiagramGroup`: The diagram group
  - `exp_idx::Int`: Index into cache.exponent_indices (1-based)

# Returns

  - `MutableBackBone`: The filled backbone (same object as input)
"""
function fill_backbone_from_group!(
        bb::MutableBackBone,
        cache::DiagramConfigCache{L},
        group::DiagramGroup,
        exp_idx::Int
) where {L}
    return fill_backbone_from_configs!(
        bb,
        cache,
        cache.topologies[group.topo_idx],
        cache.directions[group.dir_idx],
        cache.flavors[group.flav_idx],
        cache.exponent_indices[exp_idx]
    )
end
