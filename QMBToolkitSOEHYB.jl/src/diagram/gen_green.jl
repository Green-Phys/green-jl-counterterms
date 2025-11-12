# Green's function diagram

function generate_green_diagrams(expansion_order::Int, n_flavors::Int, n_flavors_physical::Int)
    
    diagrams = [] 
    if expansion_order == 0
        # |beta| --- (R) --- |c_{i}(tau)| --- (R) --- |cdag_{j}(0)| M=3
        for i in 1:n_flavors_physical
            for j in 1:n_flavors_physical
                node1 = BackBoneNode(:cdag, j, nothing, DeltaHyb(1,2))
                edge1 = BackBoneEdge([KMinus(1)])
                node2 = BackBoneNode(:c, i, nothing, NoHyb())
                edge2 = BackBoneEdge([KPlus(1), KPlus(2)])        
                node3 = BackBoneNode(:empty, nothing, nothing, KPlus(1))
                backbone = BackBone(1.0, 1.0, Int[], Int[], [node3, node2, node1], [edge2, edge1], Int[])
                push!(diagrams, backbone)
            end
        end
    else
        error("Unsupported expansion order for Green's function: $expansion_order")
    end
    return diagrams
end