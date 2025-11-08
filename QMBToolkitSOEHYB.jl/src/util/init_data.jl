using QMBToolkitED
using DelimitedFiles
using Lehmann

"""
    create_model_data(model_params::ModelParams)

Create a `ModelData` object from the parameters in a `ModelParams` struct.
This involves reading data from files to construct the Hamiltonian, operators,
and hybridization function.
"""
function create_model_data(model_params::ModelParams, dlr_params::DLRParams)
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
    @assert n_orb == n_flavors "Number of orbitals in hybridization function must match number of flavors from local Hamiltonian."
    delta_flat = Delta_data[:, 4]
    Delta_permuted = reshape(delta_flat, (n_orb, n_orb, n_tau))
    Delta = permutedims(Delta_permuted, (3, 2, 1))  # Now Delta[tau, orb_1, orb_2]
    
    ### Create DLR grid
    tau_grid_Delta = collect(LinRange(0.0, model_params.beta, n_tau))
    dlr_grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)

    return ModelData(
        model_params.beta,
        n_flavors,
        n_fock,
        ann_opts,
        cre_opts,
        evals,
        evecs,
        Delta,
        tau_grid_Delta,
        dlr_grid
    )
end

"""
    create_solver_data(model_data::ModelData, params::Params)

Create a `SolverData` object from the `ModelData` and `Params`.
"""
function create_solver_data(model_data::ModelData)
    Delta_dlr = tau2dlr(model_data.dlr_grid, model_data.Delta, model_data.tau_grid_Delta)
    Delta_reconstructed = dlr2tau(model_data.dlr_grid, Delta_dlr, model_data.tau_grid_Delta)
    abs_diff = maximum(abs.(Delta_reconstructed .- model_data.Delta))
    calculation_data = CalculationData(
        abs_diff,
        Delta_dlr,
        zeros(Float64, size(Delta_dlr)),
        zeros(Float64, size(Delta_dlr)),
        zeros(Float64, size(Delta_dlr)),
        zeros(Float64, size(Delta_dlr))
    )
    return SolverData(model_data, calculation_data)
end

