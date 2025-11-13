# Green's function diagram

function generate_green_diagrams(expansion_order::Int, n_flavors::Int, annihilation_index::Int, creation_index::Int)
    
    diagrams = [] 
    if expansion_order == 0 || expansion_order == 1 
        # 0th and 1st order Green's function diagrams are the same
        # |beta| --- (R) --- |c_{annihilation_index}(tau)| --- (R) --- |cdag_{creation_index}(0)| M=3, L=0
        node1 = BackBoneNode(:cdag, creation_index, nothing, NoHyb())
        edge1 = BackBoneEdge([])
        node2 = BackBoneNode(:c, annihilation_index, nothing, NoHyb())
        edge2 = BackBoneEdge([])        
        node3 = BackBoneNode(:empty, nothing, nothing, NoHyb())
        sign = -1.0
        backbone = BackBone(:green_diagram, sign, Int[], [node3, node2, node1], [edge2, edge1])
        push!(diagrams, backbone)
    else
        error("Unsupported expansion order for Green's function: $expansion_order")
    end
    return diagrams
end