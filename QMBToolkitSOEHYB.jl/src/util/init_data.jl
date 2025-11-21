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

    ### Load local Hamiltonian data
    H, ann_opts = QMBToolkitED.build_hamiltonian(model_params.file_path*model_params.hopping_file, model_params.file_path*model_params.interaction_file)
    ann_opts = vec([Matrix{Float64}(c) for c in ann_opts]) # convert to Float64
    n_flavors = length(ann_opts)
    n_fock = 1 << n_flavors
    cre_opts = vec([Matrix{Float64}(adjoint(c)) for c in ann_opts])
    evals, evecs = QMBToolkitED.solve_hamiltonian(H)
    
    ### init DLR grid
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
   
    ### Load hybridization function data
    hyb_poles = readdlm(model_params.file_path*model_params.hybridization_poles_file)
    hyb_poles = vec(hyb_poles)
    hyb_weights_orig = readdlm(model_params.file_path*model_params.hybridization_weights_file)
    @assert size(hyb_weights_orig, 2) == 4  # ind_poles, orb_1, orb_2, weights
    n_poles = Int(maximum(hyb_weights_orig[:, 1])) + 1
    n_flavors_physical = Int(maximum(hyb_weights_orig[:, 2:3])) + 1
    @assert size(hyb_weights_orig, 1) == n_poles * n_flavors_physical * n_flavors_physical
    hyb_weights_flat = hyb_weights_orig[:, 4]
    hyb_weights_permuted = reshape(hyb_weights_flat, (n_flavors_physical, n_flavors_physical, n_poles))
    hyb_weights_physical = permutedims(hyb_weights_permuted, (3, 2, 1))  # Now Delta[tau, orb_1, orb_2]

    # hyb_weights contains both hyb_weights physical and those for counterterms (zeros)
    hyb_weights = zeros(Float64, n_poles, n_flavors, n_flavors)
    hyb_weights[:, 1:n_flavors_physical, 1:n_flavors_physical] = hyb_weights_physical

    # Convert hybridization function to DLR representation
    KPlus_on_DLRtau = _gen_K_on_DLRtau(hyb_poles, dlr_grid)
    KMinus_on_DLRtau = _gen_K_on_DLRtau(-hyb_poles, dlr_grid)
    hyb_soe = SOEData(hyb_weights, hyb_poles, KPlus_on_DLRtau, KMinus_on_DLRtau)

    ### Derived quantities
    n_fock_physical = 1 << n_flavors_physical
    
    # Convert bare propagator e^{-tau H} to DLR representation
    bare_propagator_dlr = tau2dlr(dlr_grid, bare_propagator, dlr_grid.tau)
    bare_propagator_reconstructed = dlr2tau(dlr_grid, bare_propagator_dlr, dlr_grid.tau)
    bare_propagator_err = maximum(abs.(bare_propagator_reconstructed .- bare_propagator))


    # precompute annihilation and creation operators multiplied with SOE coefficients of the hybridization function
    ann_opts_SOE = _gen_opts_SOE(ann_opts, hyb_soe.coeffs, false)
    cre_opts_SOE = _gen_opts_SOE(cre_opts, hyb_soe.coeffs, true)

    # precompute K0_inverse
    K0_inverse = _gen_K0_inverse(hyb_poles, model_params.beta)

    return ModelData(
        model_params.beta,
        n_flavors,
        n_fock,
        ann_opts,
        cre_opts,
        evals,
        evecs,
        dlr_grid,
        E_shift,
        n_flavors_physical,
        n_fock_physical,
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
        1.0 # partition function
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
