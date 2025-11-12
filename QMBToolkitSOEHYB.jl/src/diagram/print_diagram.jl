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
            op = "c†"
        end
    end

    # Determine subindex string
    sub = ""
    if node.bar_index !== nothing
        sub = "$(node.bar_index)"
    elseif node.flavor !== nothing
        sub = "$(node.flavor)"
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
    println("Prefactor: ", backbone.prefactor)
    println()

    nodes = backbone.nodes
    edges = backbone.edges
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
    edge_len = max_half_width_sum + 3 # +3 for " R "

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
    min_start = 1
    max_end = 0
    for i in 1:M
        center = node_center_pos[i]
        node_s, op_part, hyb_part = node_infos[i]
        
        align_pos = 1
        
        start_s = center - align_pos + 1
        end_s = start_s + length(node_s) - 1
        
        if start_s < min_start
            min_start = start_s
        end
        if end_s > max_end
            max_end = end_s
        end
    end
    
    # Adjust all positions if we have negative start
    if min_start < 1
        shift = 1 - min_start
        node_center_pos .+= shift
        max_end += shift
    end
    total_width = max_end

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
        center = node_center_pos[i]
        line3[center] = '│'
        
        node_s, op_part, hyb_part = node_infos[i]
        
        align_pos = 1
        
        start_s = center - align_pos + 1
        end_s = start_s + length(node_s) - 1
        line2[start_s:end_s] = collect(node_s)
    end

    # Draw edges and R
    for i in 1:M-1
        start = node_center_pos[i] + 1
        stop = node_center_pos[i+1] - 1
        if start <= stop
            line3[start:stop] .= '─'
            
            mid = (start + stop) ÷ 2
            line4[mid] = 'R'
        end
    end

    # Draw hybridization lines
    for k in 1:max_hybs
        for i in 1:M-1
            if k <= length(edges[i].attached_hybs)
                hyb = edges[i].attached_hybs[k]
                hyb_str = format_hyb(hyb)
                
                edge_start = node_center_pos[i] + 1
                edge_stop = node_center_pos[i+1] - 1
                
                if edge_start <= edge_stop
                    edge_mid = (edge_start + edge_stop) ÷ 2
                    
                    underscore_pos = findfirst('_', hyb_str)
                    if underscore_pos !== nothing
                        start_h = edge_mid - underscore_pos + 3
                    else
                        # Fallback to center alignment
                        start_h = edge_mid - (length(hyb_str) ÷ 2)
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
    println()
    println(String(line3))
    println(String(line4))
    for line in hyb_lines
        println(String(line))
    end
end
