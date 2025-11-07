using QMBToolkitED
using DelimitedFiles

# fill information about hybridization function and local Hamiltonian in ModelParams

function fill_model_params!(model::ModelParams)
    ### Load local Hamiltonian data
    H, model.ann_opts = QMBToolkitED.build_hamiltonian(model.hopping_file, model.interaction_file)
    model.n_flavors = length(model.ann_opts)
    model.n_fock = 1 << model.n_flavors
    model.cre_opts = [adjoint(c) for c in model.ann_opts]
    model.evals, model.evecs = QMBToolkitED.solve_hamiltonian(H)
    ### Load hybridization function data
    Delta_data = readdlm(model.hybridization_file)
    @assert size(Delta_data, 2) == 5  # ind_tau, orb_1, orb_2, ReDelta, ImDelta
    n_orb = Int(maximum(Delta_data[:, 2:3])) + 1
    n_tau = Int(maximum(Delta_data[:, 1])) + 1
    @assert size(Delta_data, 1) == n_tau * n_orb * n_orb
    delta_flat = Delta_data[:, 4]
    Delta_permuted = reshape(delta_flat, (n_orb, n_orb, n_tau)) # (orb_1, orb_2, tau)
    model.Delta = permutedims(Delta_permuted, (3, 2, 1)) 
end
