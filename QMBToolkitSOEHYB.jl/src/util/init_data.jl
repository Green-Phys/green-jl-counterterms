using QMBToolkitED
using DelimitedFiles
using QMBToolkitDLR
using LinearAlgebra

"""
    init_model_data(model_params::ModelParams)

init a `ModelData` object from the parameters in a `ModelParams` struct.
This involves reading data from files to construct the Hamiltonian, operators,
and hybridization function.
"""
function init_model_data(model_params::ModelParams, dlr_params::DLRParams)
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
    ann_opts = vec([Matrix{Float64}(c) for c in ann_opts]) # convert to Float64
    n_flavors = length(ann_opts)
    n_fock = 1 << n_flavors
    cre_opts = vec([Matrix{Float64}(adjoint(c)) for c in ann_opts])
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
    hyb_input = permutedims(hyb_permuted, (3, 2, 1))  # Now Delta[tau, orb_1, orb_2]
    
    ### init DLR grid
    tau_grid_hyb = collect(LinRange(0.0, model_params.beta, n_tau))
    dlr_grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)
    
    ### init bare propagator
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


    # Convert hybridization function to DLR representation
    hyb_dlr = tau2dlr(dlr_grid, hyb_input, tau_grid_hyb)
    hyb_reconstructed = dlr2tau(dlr_grid, hyb_dlr, tau_grid_hyb)
    hyb_error = maximum(abs.(hyb_reconstructed .- hyb_input))
    hyb_tau = dlr2tau(dlr_grid, hyb_dlr, dlr_grid.tau)

    hyb_on_DLR = DLRData(hyb_tau, hyb_dlr)
    hyb_coeffs = hyb_dlr
    hyb_exponents = dlr_grid.omega
    KPlus_on_DLRtau = _gen_K_on_DLRtau(hyb_exponents, dlr_grid)
    KMinus_on_DLRtau = _gen_K_on_DLRtau(-hyb_exponents, dlr_grid)
    hyb_soe = SOEData(hyb_coeffs, hyb_exponents, KPlus_on_DLRtau, KMinus_on_DLRtau, hyb_on_DLR)

    val = _get_val_from_SOE(hyb_soe, dlr_grid.tau, model_params.beta)
    @assert maximum(abs.(val .- hyb_tau)) < 1e-14
    
    # Convert bare propagator e^{-tau H} to DLR representation
    bare_propagator_dlr = tau2dlr(dlr_grid, bare_propagator, dlr_grid.tau)
    bare_propagator_reconstructed = dlr2tau(dlr_grid, bare_propagator_dlr, dlr_grid.tau)
    bare_propagator_err = maximum(abs.(bare_propagator_reconstructed .- bare_propagator))


    # precompute annihilation and creation operators multiplied with SOE coefficients of the hybridization function
    ann_opts_SOE = _gen_opts_SOE(ann_opts, hyb_soe.coeffs, false)
    cre_opts_SOE = _gen_opts_SOE(cre_opts, hyb_soe.coeffs, true)

    # precompute K0_inverse
    K0_inverse = _gen_K0_inverse(dlr_grid.omega, model_params.beta)

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
        hyb_input,
        tau_grid_hyb,
        dlr_grid,
        E_shift,
        n_flavors_physical,
        n_fock_physical,
        hyb_error,
        bare_propagator_err,
        hyb_soe,
        DLRData(bare_propagator, bare_propagator_dlr), # bare propagator
        K0_inverse,
        ann_opts_SOE,
        cre_opts_SOE
    )
end

"""
    init_solver_data(model_data::ModelData, params::Params)

init a `SolverData` object from the `ModelData` and `Params`.
"""
function init_solver_data(model_data::ModelData)

    calculation_data = CalculationData(
        DLRData(model_data.dlr_grid.size, model_data.n_fock, model_data.n_fock), # pseudo-particle propagator
        DLRData(model_data.dlr_grid.size, model_data.n_fock, model_data.n_fock), # pseudo-particle self-energy
        DLRData(model_data.dlr_grid.size, model_data.n_flavors_physical, model_data.n_flavors_physical), # physical Green's function
    )
    return SolverData(model_data, calculation_data)
end

function _gen_K0_inverse(exponents::Vector{Float64},beta::Float64)
    K0_inverse = zeros(Float64, length(exponents))
    for (i, omega) in enumerate(exponents)
        if omega <=0
            K0_inverse[i] = 1.0 + exp(beta * omega)
        else
            K0_inverse[i] = 1.0 + exp(-beta * omega)
        end
    end
    # all values should between 1 and 2
    @assert all(K0_inverse .>= 1.0) && all(K0_inverse .<= 2.0) "K0_inverse values should be between 1 and 2."
    return K0_inverse
end

function _gen_K_on_DLRtau(exponents::Vector{Float64}, dlr_grid::DLRGrid)
    tau_grid = dlr_grid.tau
    K_on_DLRtau = zeros(Float64, length(tau_grid), length(exponents))
    for (j, omega) in enumerate(exponents)
        for (i, tau) in enumerate(tau_grid)
            K_on_DLRtau[i, j] = kernel_tau(tau, omega, dlr_grid.beta)
        end
    end
    return K_on_DLRtau
end

function _get_val_from_SOE(soe::SOEData, tau_grid::Vector{Float64}, beta::Float64)
    # sum_{i} coeffs[i] * K(tau, exponents[i]), different matrix element share the same exponents
    n_tau = length(tau_grid)
    p = size(soe.coeffs, 1)
    o1 = size(soe.coeffs, 2)
    o2 = size(soe.coeffs, 3)
    val = zeros(Float64, n_tau, o1, o2)
    for (k, tau) in enumerate(tau_grid)
        for j in 1:p
            K_val = kernel_tau(tau, soe.exponents[j], beta)
            val[k, :, :] .+= soe.coeffs[j, :, :] * K_val
        end
    end
    return val
end


function _gen_opts_SOE(opts::Vector{Matrix{Float64}}, coeffs::Array{Float64,3}, is_creation::Bool)
    # coeffs shape: (p, o1, o2), o1 correspond to creation and o2 correspond to annihilation
    # \bar{F}^{\dagger}_{o, l} = \sum_{o'} Delta_{l, o', o} F^{\dagger}_{o'}
    # \bar{F}_{o, l} = \sum_{o'} Delta_{l, o, o'} F_{o'}

    n_flavors = length(opts)
    p = size(coeffs, 1)
    @assert size(coeffs, 2) == n_flavors "Size mismatch between coeffs and number of flavors."
    @assert size(coeffs, 3) == n_flavors "Size mismatch between coeffs and number of flavors."
    opts_SOE = Matrix{Matrix{Float64}}(undef, n_flavors, p)
    for o in 1:n_flavors
        for l in 1:p
            mat = zeros(Float64, size(opts[1])...)
            if is_creation
                for o_prime in 1:n_flavors
                    mat .+= coeffs[l, o_prime, o] * opts[o_prime]
                end 
            else
                for o_prime in 1:n_flavors
                    mat .+= coeffs[l, o, o_prime] * opts[o_prime]
                end
            end
            opts_SOE[o, l] = mat
        end
    end
    return opts_SOE
end
