"""
Common backbone diagram evaluation utilities.

Format-agnostic functions that work with both Dense and BlockDense representations
using the interface functions defined in format/interface.jl.
"""

"""
    get_operator(node::BackBoneNode, model_data::ModelData)

Get the operator for a given node from model data.

This is a format-agnostic function that uses interface accessors to retrieve
the appropriate operator (Dense or BlockDense) based on the model_data type.

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
