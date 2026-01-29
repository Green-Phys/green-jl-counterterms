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
    check_flux_sequence(topo_config, directions, flavors, basis, diagram_type) -> Bool

Check if the operator sequence produces only valid intermediate fluxes.

Returns true if all intermediate fluxes encountered during backbone evaluation
would have non-empty block structure in the given basis.

# Arguments

  - `topo_config::TopologyConfig`: Topology configuration
  - `directions::NTuple{L,Symbol}`: Direction for each hybridization line
  - `flavors::NTuple{L,Int}`: Flavor index for each hybridization line
  - `basis::Basis`: The basis defining valid fluxes
  - `diagram_type::Symbol`: :propagator_diagram or :green_diagram

# Returns

  - `Bool`: true if all intermediate fluxes are valid, false otherwise
"""
function check_flux_sequence(
        topo_config::TopologyConfig,
        directions::NTuple{L, Symbol},
        flavors::NTuple{L, Int},
        basis::Basis,
        diagram_type::Symbol
) where {L}
    # Get quantum number type and compute valid fluxes
    QN = typeof(basis.sectors[1].qn)
    valid_fluxes = Set{QN}()
    for src_sector in basis.sectors
        for tgt_sector in basis.sectors
            flux = tgt_sector.qn + (-src_sector.qn)
            push!(valid_fluxes, flux)
        end
    end

    # Build operator sequence with their fluxes
    # Nodes are processed in order during backbone evaluation (right to left for forward)
    M = topo_config.M

    # Create array of (node_position, operator_flux) for non-empty nodes
    # We need to track which nodes have operators and what their flux is
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

    # Sort by node position (backbone evaluation processes in specific order)
    # For forward backbone: processes from right to left (high index to low)
    # But the flux accumulation happens as we encounter operators
    sort!(node_fluxes, by = x -> x[1])

    # Simulate flux evolution
    # Start from zero flux (bare propagator has zero flux)
    current_flux = zero(QN)

    # For propagator diagrams: evaluate right to left, so reverse order
    # For green diagrams: similar pattern
    # The key insight: we process operators in the order they appear in the backbone
    # which is determined by the pairing structure

    # Actually, the backbone evaluation in multiplication_convolution processes
    # nodes[i+1] for i in 1:(length(edges)-1), which is nodes[2], nodes[3], ...
    # So we need to simulate this order

    # For propagator: nodes 1 and M are empty (boundary)
    # For green: nodes 1, annihilation_pos, and M are empty

    for (node_pos, op_flux) in node_fluxes
        # Skip boundary nodes (they are :empty)
        if diagram_type == :propagator_diagram
            if node_pos == 1 || node_pos == M
                continue
            end
        else  # :green_diagram
            if node_pos == 1 || node_pos == M || node_pos == topo_config.annihilation_pos
                continue
            end
        end

        # Apply operator flux
        current_flux = current_flux + op_flux

        # Check if this intermediate flux is valid
        if current_flux ∉ valid_fluxes
            return false
        end
    end

    return true
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
    groups = DiagramGroup[]

    for topo_idx in 1:cache.n_topologies
        topo_config = cache.topologies[topo_idx]

        for dir_idx in 1:cache.n_directions
            directions = cache.directions[dir_idx]

            for flav_idx in 1:cache.n_flavors_combo
                flavors = cache.flavors[flav_idx]

                # Check if this combination has valid flux sequence
                if check_flux_sequence(topo_config, directions, flavors, basis, cache.diagram_type)
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
