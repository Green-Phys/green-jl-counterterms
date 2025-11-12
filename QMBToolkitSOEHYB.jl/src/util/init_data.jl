using QMBToolkitED
using DelimitedFiles
using QMBToolkitDLR
using LinearAlgebra

"""
    create_model_data(model_params::ModelParams)

Create a `ModelData` object from the parameters in a `ModelParams` struct.
This involves reading data from files to construct the Hamiltonian, operators,
and hybridization function.
"""
function create_model_data(model_params::ModelParams, dlr_params::DLRParams)
    ### Load pole energies and hybridizations, the file could be empty
    poles_energies = Float64[]
    poles_hybs = zeros(Float64, 0, 0)
    if isfile(model_params.file_path*model_params.poles_energies_file) && filesize(model_params.file_path*model_params.poles_energies_file) > 0
        poles_energies = readdlm(model_params.file_path*model_params.poles_energies_file)[:, 1]
    end
    if isfile(model_params.file_path*model_params.poles_hybs_file) && filesize(model_params.file_path*model_params.poles_hybs_file) > 0
        poles_hybs = readdlm(model_params.file_path*model_params.poles_hybs_file)
        @assert size(poles_hybs, 1) == length(poles_energies) "Number of rows in poles_hybs must match number of pole energies."
    end

    ### Load local Hamiltonian data
    H, ann_opts = QMBToolkitED.build_hamiltonian(model_params.file_path*model_params.hopping_file, model_params.file_path*model_params.interaction_file)
    n_flavors = length(ann_opts)
    n_fock = 1 << n_flavors
    cre_opts = [adjoint(c) for c in ann_opts]
    evals, evecs = QMBToolkitED.solve_hamiltonian(H)

    ### Load hybridization function data
    hyb_data = readdlm(model_params.file_path*model_params.hybridization_file)
    @assert size(hyb_data, 2) == 5  # ind_tau, orb_1, orb_2, ReDelta, ImDelta
    n_orb = Int(maximum(hyb_data[:, 2:3])) + 1
    n_tau = Int(maximum(hyb_data[:, 1])) + 1
    @assert size(hyb_data, 1) == n_tau * n_orb * n_orb
    @assert n_orb == n_flavors "Number of orbitals in hybridization function must match number of flavors from local Hamiltonian."
    hyb_flat = hyb_data[:, 4]
    hyb_permuted = reshape(hyb_flat, (n_orb, n_orb, n_tau))
    hyb = permutedims(hyb_permuted, (3, 2, 1))  # Now Delta[tau, orb_1, orb_2]
    
    ### Create DLR grid
    tau_grid_hyb = collect(LinRange(0.0, model_params.beta, n_tau))
    dlr_grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)
    
    ### create bare propagator
    tau_grid = dlr_grid.tau
    bare_propagator = zeros(Float64, length(tau_grid), n_fock, n_fock)
    E_shift = minimum(evals)
    evals_shifted = evals .- E_shift
    for (k, tau) in enumerate(tau_grid)
        U_tau = exp.(-tau .* evals_shifted)
        bare_propagator[k, :, :] = evecs * Diagonal(U_tau) * adjoint(evecs)
    end

    ### Derived quantities
    n_flavors_physical = n_flavors - length(poles_energies)
    n_fock_physical = 1 << n_flavors_physical

    return ModelData(
        poles_energies,
        poles_hybs,
        model_params.beta,
        n_flavors,
        n_fock,
        ann_opts,
        cre_opts,
        evals,
        evecs,
        hyb,
        tau_grid_hyb,
        dlr_grid,
        bare_propagator,
        E_shift,
        n_flavors_physical,
        n_fock_physical
    )
end

"""
    create_solver_data(model_data::ModelData, params::Params)

Create a `SolverData` object from the `ModelData` and `Params`.
"""
function create_solver_data(model_data::ModelData)
    # Convert hybridization function to DLR representation
    hyb_dlr = tau2dlr(model_data.dlr_grid, model_data.hyb, model_data.tau_grid_hyb)
    hyb_reconstructed = dlr2tau(model_data.dlr_grid, hyb_dlr, model_data.tau_grid_hyb)
    hyb_error = maximum(abs.(hyb_reconstructed .- model_data.hyb))
    hyb_tau = dlr2tau(model_data.dlr_grid, hyb_dlr, model_data.dlr_grid.tau)
    
    # Convert bare propagator e^{-tau H} to DLR representation
    bare_propagator_dlr = tau2dlr(model_data.dlr_grid, model_data.bare_propagator, model_data.dlr_grid.tau)
    bare_propagator_reconstructed = dlr2tau(model_data.dlr_grid, bare_propagator_dlr, model_data.dlr_grid.tau)
    bare_propagator_err = maximum(abs.(bare_propagator_reconstructed .- model_data.bare_propagator))

    calculation_data = CalculationData(
        hyb_error,
        bare_propagator_err,
        DLRData(hyb_tau, hyb_dlr), # hybridization function
        DLRData(model_data.bare_propagator, bare_propagator_dlr), # bare propagator
        DLRData(model_data.bare_propagator, bare_propagator_dlr), # bold propagator 
        DLRData(model_data.dlr_grid.size, model_data.n_flavors,model_data.n_flavors), # pseudo-particle self-energy
        DLRData(model_data.dlr_grid.size, model_data.n_flavors_physical, model_data.n_flavors_physical) # physical Green's function
    )
    return SolverData(model_data, calculation_data)
end

