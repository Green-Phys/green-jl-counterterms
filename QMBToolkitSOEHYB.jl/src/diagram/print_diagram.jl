function format_node(node::BackBoneNode)
    # Determine operator string
    op = ""
    if node.bar_index !== nothing
        op = "c̄"
        if node.opt_type == :cdag
            op *= "†"
        end
    else
        if node.opt_type == :c
            op = "c"
        elseif node.opt_type == :cdag
            op = "c"*"†"
        end
    end    

    # Determine subindex string
    sub = ""
    if node.flavor !== nothing
        sub = "$(node.flavor)"
    end
    if node.bar_index !== nothing
        sub *= "$(node.bar_index)"
    end

    op_part = op * (isempty(sub) ? "" : "_{$sub}")

    # Determine hybridization part
    hyb_part = ""
    if node.attached_hyb isa KPlus
        hyb_part = "K⁺_{$(node.attached_hyb.l)}"
    elseif node.attached_hyb isa KMinus
        hyb_part = "K⁻_{$(node.attached_hyb.l)}"
    elseif node.attached_hyb isa DeltaHyb
        hyb_part = "Δ_{$(node.attached_hyb.i),$(node.attached_hyb.j)}"
    end
    
    return (op_part * hyb_part, op_part, hyb_part)
end

function format_hyb(hyb::HybridizationState)
    if hyb isa KPlus
        return "K⁺_{$(hyb.l)}"
    elseif hyb isa KMinus
        return "K⁻_{$(hyb.l)}"
    else
        return ""
    end
end

function print_diagram(backbone::BackBone)
    println("Diagram type: ", backbone.diagram_type)
    println("Sign: ", backbone.sign)
    println("Prefactor K exponents: ", backbone.prefactor_Kexps)

    nodes = reverse(backbone.nodes)
    edges = reverse(backbone.edges)
    M = length(nodes)

    if M == 0
        return
    end

    node_infos = [format_node(n) for n in nodes]
    node_strs = [info[1] for info in node_infos]
    node_widths = [max(1, length(s)) for s in node_strs]
    
    # Determine constant edge length
    max_half_width_sum = 0
    if M > 1
        for i in 1:M-1
            half_sum = (node_widths[i] ÷ 2) + (node_widths[i+1] ÷ 2)
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
    
    # Calculate total width needed
    min_overall_coord = 1 # Initialize with 1, as coordinates are 1-based
    max_overall_coord = 0
    for i in 1:M
        center = node_center_pos[i]
        node_s, op_part, _ = node_infos[i]
        
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

    max_hybs = 0
    if !isempty(edges)
        max_hybs = maximum(length(e.attached_hybs) for e in edges)
    end

    # Create char arrays for lines
    line2 = fill(' ', total_width)
    line3 = fill(' ', total_width)
    line4 = fill(' ', total_width)
    hyb_lines = [fill(' ', total_width) for _ in 1:max_hybs]

    # Place nodes and markers
    for i in 1:M
        center_unshifed = node_center_pos[i]
        center_on_line = center_unshifed - min_overall_coord + 1
        line3[center_on_line] = '│'
        
        node_s = node_strs[i]
        tot_part, opt_part, hyb_part = node_infos[i]
        align_pos = (length(tot_part) + 1) ÷ 2 -1
        if hyb_part == ""
            # change align_post to the underscore position in opt_part
            underscore_pos = findfirst(isequal('_'), collect(opt_part))
            if underscore_pos !== nothing
                align_pos = underscore_pos-1
            end
        end
        
        start_s_unshifed = center_unshifed - align_pos + 1
        start_s_on_line = start_s_unshifed - min_overall_coord + 1
        end_s_on_line = start_s_on_line + length(node_s) - 1
        
        line2[start_s_on_line:end_s_on_line] = collect(node_s)
    end

    # Draw edges and R
    for i in 1:M-1
        start_unshifed = node_center_pos[i] + 1
        stop_unshifed = node_center_pos[i+1] - 1
        
        start_on_line = start_unshifed - min_overall_coord + 1
        stop_on_line = stop_unshifed - min_overall_coord + 1

        if start_on_line <= stop_on_line
            line3[start_on_line:stop_on_line] .= '─'
            
            mid_on_line = (start_on_line + stop_on_line) ÷ 2
            if backbone.diagram_type == :propagator_diagram
                line4[mid_on_line:mid_on_line+1] = collect("R0")
            else
                line4[mid_on_line] = 'R'
            end
        end
    end

    # Draw hybridization lines
    for k in 1:max_hybs
        for i in 1:M-1
            if k <= length(edges[i].attached_hybs)
                hyb = edges[i].attached_hybs[k]
                hyb_str = format_hyb(hyb)
                
                edge_start_unshifed = node_center_pos[i] + 1
                edge_stop_unshifed = node_center_pos[i+1] - 1
                
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

    # Print lines
    println(String(line2))
    println(String(line3))
    println(String(line4))
    for line in hyb_lines
        println(String(line))
    end
end
