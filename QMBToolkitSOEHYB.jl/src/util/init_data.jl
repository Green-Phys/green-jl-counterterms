using QMBToolkitED
using DelimitedFiles
using Lehmann

"""
    create_model_data(model_params::ModelParams)

Create a `ModelData` object from the parameters in a `ModelParams` struct.
This involves reading data from files to construct the Hamiltonian, operators,
and hybridization function.
"""
function create_model_data(model_params::ModelParams)
    ### Load local Hamiltonian data
    H, ann_opts = QMBToolkitED.build_hamiltonian(model_params.hopping_file, model_params.interaction_file)
    n_flavors = length(ann_opts)
    n_fock = 1 << n_flavors
    cre_opts = [adjoint(c) for c in ann_opts]
    evals, evecs = QMBToolkitED.solve_hamiltonian(H)

    ### Load hybridization function data
    Delta_data = readdlm(model_params.hybridization_file)
    @assert size(Delta_data, 2) == 5  # ind_tau, orb_1, orb_2, ReDelta, ImDelta
    n_orb = Int(maximum(Delta_data[:, 2:3])) + 1
    n_tau = Int(maximum(Delta_data[:, 1])) + 1
    @assert size(Delta_data, 1) == n_tau * n_orb * n_orb
    delta_flat = Delta_data[:, 4]
    Delta_permuted = reshape(delta_flat, (n_orb, n_orb, n_tau)) # (orb_1, orb_2, tau)
    Delta = permutedims(Delta_permuted, (3, 2, 1)) 

    return ModelData(
        n_flavors,
        n_fock,
        ann_opts,
        cre_opts,
        evals,
        evecs,
        Delta
    )
end

"""
    create_dlr_data(dlr_params::DLRParams, model_params::ModelParams)

Create a `DLRData` object containing the DLR grid.
"""
function create_dlr_data(dlr_params::DLRParams, model_params::ModelParams)
    grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)
    return DLRData(grid)
end
