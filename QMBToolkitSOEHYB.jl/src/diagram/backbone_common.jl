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
    # Use interface accessors - they dispatch based on format
    cre_opts = get_cre_opts(model_data)
    ann_opts = get_ann_opts(model_data)
    cre_opts_bar = get_cre_opts_bar(model_data)
    ann_opts_bar = get_ann_opts_bar(model_data)

    if node.opt_type == :cdag && node.bar_index == -1
        return cre_opts[node.flavor]
    elseif node.opt_type == :c && node.bar_index == -1
        return ann_opts[node.flavor]
    elseif node.opt_type == :cdag && node.bar_index != -1
        return cre_opts_bar[node.flavor, node.bar_index]
    elseif node.opt_type == :c && node.bar_index != -1
        return ann_opts_bar[node.flavor, node.bar_index]
    end
end
