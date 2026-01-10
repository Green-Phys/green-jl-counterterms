function format_node(node::BackBoneNode, is_external_opt::Bool = false, is_external_ann::Bool = false)
    # Determine operator string
    op = ""
    if is_external_opt
        if is_external_ann
            op = "c"
        else
            op = "c⁺"
        end
    else
        if node.bar_index != -1 && node.opt_type == :c
            op = "F̅"
        elseif node.bar_index != -1 && node.opt_type == :cdag
            op = "F̅⁺"
        elseif node.opt_type == :c
            op = "F"
        elseif node.opt_type == :cdag
            op = "F⁺"
        elseif node.opt_type == :empty
            op = ""
        else
            error("Unrecognized operator type")
        end
    end

    # Determine subindex string
    sub = ""
    if node.flavor != -1
        sub = "$(node.flavor)"
    end
    if node.bar_index != -1
        sub *= ",$(node.bar_index)"
    end

    op_part = op * (isempty(sub) ? "" : "_{$sub}")

    # Determine hybridization part
    hyb_part = ""
    if is_kplus(node.attached_hyb)
        hyb_part = "K⁺_{$(get_kplus_index(node.attached_hyb))}"
    elseif is_kminus(node.attached_hyb)
        hyb_part = "K⁻_{$(get_kminus_index(node.attached_hyb))}"
    end

    return (op_part * hyb_part, op_part, hyb_part)
end

function format_hyb(hyb::HybridizationState)
    if is_kplus(hyb)
        return "K⁺_{$(get_kplus_index(hyb))}"
    elseif is_kminus(hyb)
        return "K⁻_{$(get_kminus_index(hyb))}"
    else
        return ""
    end
end

"""
    calculate_node_positions(node_widths::Vector{Int})

Calculate the center positions for each node in the diagram.
Returns (node_center_pos, edge_len) where edge_len is the distance between node centers.
"""
function calculate_node_positions(node_widths::Vector{Int})
    M = length(node_widths)

    # Determine constant edge length
    max_half_width_sum = 0
    if M > 1
        for i in 1:(M - 1)
            half_sum = (node_widths[i] ÷ 2) + (node_widths[i + 1] ÷ 2)
            if half_sum > max_half_width_sum
                max_half_width_sum = half_sum
            end
        end
    end
    edge_len = max_half_width_sum + 5 # +3 for " R "
    dist_between_nodes = edge_len + 1

    # Calculate positions
    node_center_pos = zeros(Int, M)
    current_pos = 1
    # Place first node center, considering padding for its string
    current_pos += node_widths[1] ÷ 2
    node_center_pos[1] = current_pos

    for i in 2:M
        current_pos += dist_between_nodes
        node_center_pos[i] = current_pos
    end

    return (node_center_pos, edge_len)
end

"""
    calculate_total_width(node_center_pos::Vector{Int}, node_infos)

Calculate the total width needed for the diagram and adjust positions if needed.
Returns (node_center_pos, total_width, min_overall_coord).
"""
function calculate_total_width(node_center_pos::Vector{Int}, node_infos)
    M = length(node_center_pos)
    node_center_pos = copy(node_center_pos)  # Don't modify input

    # Calculate total width needed
    min_overall_coord = 1 # Initialize with 1, as coordinates are 1-based
    max_overall_coord = 0
    for i in 1:M
        center = node_center_pos[i]
        node_s, _, _ = node_infos[i]

        align_pos = 1 # Align first character

        start_s = center - align_pos + 1
        end_s = start_s + length(node_s) - 1

        # Consider all coordinates that will be accessed
        min_current_coord = min(start_s, center)
        max_current_coord = max(end_s, center)

        if min_current_coord < min_overall_coord
            min_overall_coord = min_current_coord
        end
        if max_current_coord > max_overall_coord
            max_overall_coord = max_current_coord
        end
    end

    # Adjust all positions if we have negative start
    shift = 0
    if min_overall_coord < 1
        shift = 1 - min_overall_coord
        node_center_pos .+= shift # Shift node centers
        max_overall_coord += shift # Shift max overall coord
    end
    total_width = max_overall_coord

    return (node_center_pos, total_width, min_overall_coord)
end

"""
    draw_nodes!(line2, line3, node_center_pos, node_infos, node_strs, min_overall_coord)

Draw nodes and their markers on the diagram lines.
"""
function draw_nodes!(line2::Vector{Char}, line3::Vector{Char},
        node_center_pos::Vector{Int}, node_infos,
        node_strs::Vector{String}, min_overall_coord::Int)
    M = length(node_center_pos)
    for i in 1:M
        center_unshifed = node_center_pos[i]
        center_on_line = center_unshifed - min_overall_coord + 1
        line3[center_on_line] = '│'

        node_s = node_strs[i]
        tot_part, opt_part, hyb_part = node_infos[i]
        align_pos = (length(tot_part) + 1) ÷ 2 - 1
        if hyb_part == ""
            # change align_post to the underscore position in opt_part
            underscore_pos = findfirst(isequal('c'), collect(opt_part))
            if underscore_pos !== nothing
                align_pos = underscore_pos
            end
        end

        start_s_unshifed = center_unshifed - align_pos + 1
        start_s_on_line = start_s_unshifed - min_overall_coord + 1
        end_s_on_line = start_s_on_line + length(node_s) - 1

        line2[start_s_on_line:end_s_on_line] = collect(node_s)
    end
end

"""
    draw_edges!(line3, line4, node_center_pos, min_overall_coord, diagram_type)

Draw edges and R markers on the diagram lines.
"""
function draw_edges!(line3::Vector{Char}, line4::Vector{Char},
        node_center_pos::Vector{Int}, min_overall_coord::Int,
        diagram_type::Symbol)
    M = length(node_center_pos)
    for i in 1:(M - 1)
        start_unshifed = node_center_pos[i] + 1
        stop_unshifed = node_center_pos[i + 1] - 1

        start_on_line = start_unshifed - min_overall_coord + 1
        stop_on_line = stop_unshifed - min_overall_coord + 1

        if start_on_line <= stop_on_line
            line3[start_on_line:stop_on_line] .= '─'

            mid_on_line = (start_on_line + stop_on_line) ÷ 2
            if diagram_type == :propagator_diagram
                line4[mid_on_line:(mid_on_line + 1)] = collect("R0")
            else
                line4[mid_on_line] = 'R'
            end
        end
    end
end

"""
    draw_hybridizations!(hyb_lines, edges, node_center_pos, min_overall_coord, total_width)

Draw hybridization labels on the diagram lines.
"""
function draw_hybridizations!(hyb_lines::Vector{Vector{Char}}, edges::Vector{BackBoneEdge},
        node_center_pos::Vector{Int}, min_overall_coord::Int,
        total_width::Int)
    max_hybs = length(hyb_lines)
    M = length(node_center_pos)

    for k in 1:max_hybs
        for i in 1:(M - 1)
            if k <= length(edges[i].attached_hybs)
                hyb = edges[i].attached_hybs[k]
                hyb_str = format_hyb(hyb)

                edge_start_unshifed = node_center_pos[i] + 1
                edge_stop_unshifed = node_center_pos[i + 1] - 1

                if edge_start_unshifed <= edge_stop_unshifed
                    edge_mid_unshifed = (edge_start_unshifed + edge_stop_unshifed) ÷ 2
                    edge_mid_on_line = edge_mid_unshifed - min_overall_coord + 1

                    underscore_pos = findfirst('_', hyb_str)
                    start_h = if underscore_pos !== nothing
                        edge_mid_on_line - underscore_pos + 3
                    else
                        edge_mid_on_line - (length(hyb_str) ÷ 2)
                    end
                    start_h = max(1, start_h)
                    end_h = start_h + length(hyb_str) - 1
                    if end_h <= total_width
                        hyb_lines[k][start_h:end_h] = collect(hyb_str)
                    end
                end
            end
        end
    end
end

function print_diagram(backbone::BackBone)
    println("Diagram type: ", backbone.diagram_type)
    println("Sign: ", backbone.sign)
    println("Prefactor K exponents: ", backbone.prefactor_Kexps)

    # Prepare node information
    nodes = reverse(backbone.nodes)
    edges = reverse(backbone.edges)
    M = length(nodes)

    if M == 0
        return
    end

    external_opts_loc = isempty(backbone.external_opts_loc) ? Int[] :
                        [M-i+1 for i in backbone.external_opts_loc]
    node_infos = [format_node(
                      nodes[i],
                      !isempty(external_opts_loc) && i in external_opts_loc,
                      !isempty(external_opts_loc) && i == external_opts_loc[2],
                  ) for i in 1:M]
    node_strs = [info[1] for info in node_infos]
    node_widths = [max(1, length(s)) for s in node_strs]

    # Calculate layout
    node_center_pos, _ = calculate_node_positions(node_widths)
    node_center_pos, total_width,
    min_overall_coord = calculate_total_width(node_center_pos, node_infos)

    # Determine maximum hybridizations per edge
    max_hybs = 0
    if !isempty(edges)
        max_hybs = maximum(length(e.attached_hybs) for e in edges)
    end

    # Create char arrays for lines
    line2 = fill(' ', total_width)
    line3 = fill(' ', total_width)
    line4 = fill(' ', total_width)
    hyb_lines = [fill(' ', total_width) for _ in 1:max_hybs]

    # Draw diagram components
    draw_nodes!(line2, line3, node_center_pos, node_infos, node_strs, min_overall_coord)
    draw_edges!(line3, line4, node_center_pos, min_overall_coord, backbone.diagram_type)
    draw_hybridizations!(hyb_lines, edges, node_center_pos, min_overall_coord, total_width)

    # Print lines
    println(String(line2))
    println(String(line3))
    println(String(line4))
    for line in hyb_lines
        println(String(line))
    end
end
