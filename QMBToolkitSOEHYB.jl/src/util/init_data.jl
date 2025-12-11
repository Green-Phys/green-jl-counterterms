using QMBToolkitED
using DelimitedFiles
using QMBToolkitDLR
using LinearAlgebra

# Constants for file format validation
const HYB_WEIGHTS_EXPECTED_COLS = 4  # Expected columns: ind_poles, orb_1, orb_2, weights

"""
    init_model_data(model_params::ModelParams)

init a `ModelData` object from the parameters in a `ModelParams` struct.
This involves reading data from files to construct the Hamiltonian, operators,
and hybridization function.
"""
function init_model_data(model_params::ModelParams, dlr_params::DLRParams)

    ### Load local Hamiltonian data
    hopping_path = joinpath(model_params.file_path, model_params.hopping_file)
    interaction_path = joinpath(model_params.file_path, model_params.interaction_file)
    isfile(hopping_path) || error("Hopping file not found: $hopping_path")
    isfile(interaction_path) || error("Interaction file not found: $interaction_path")

    H, ann_opts_raw = QMBToolkitED.build_hamiltonian(hopping_path, interaction_path)
    evals, evecs = QMBToolkitED.solve_hamiltonian(H)

    # Infer numeric type from eigenvectors (Float64 or ComplexF64)
    T = eltype(evecs)

    # Convert operators to the correct type
    ann_opts = vec([convert(Matrix{T}, c) for c in ann_opts_raw])
    n_flavors = length(ann_opts)
    n_fock = 1 << n_flavors
    cre_opts = vec([Matrix{T}(adjoint(c)) for c in ann_opts])
    
    ### init DLR grid
    dlr_grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)
    
    ### init bare propagator
    tau_grid = dlr_grid.tau
    bare_propagator = zeros(T, length(tau_grid), n_fock, n_fock)
    E_shift = minimum(evals)
    evals_shifted = evals .- E_shift
    for (k, tau) in enumerate(tau_grid)
        U_tau = exp.(-tau * evals_shifted)
        # Compute propagator: evecs * exp(-tau*H) * evecs^†
        # For Hermitian H, this should be Hermitian, but may have small imaginary parts
        bare_propagator[k, :, :] = evecs * Diagonal(U_tau) * adjoint(evecs)
    end
   
    ### Load hybridization function data
    hyb_poles_path = joinpath(model_params.file_path, model_params.hybridization_poles_file)
    hyb_weights_path = joinpath(model_params.file_path, model_params.hybridization_weights_file)
    isfile(hyb_poles_path) || error("Hybridization poles file not found: $hyb_poles_path")
    isfile(hyb_weights_path) || error("Hybridization weights file not found: $hyb_weights_path")

    hyb_poles = readdlm(hyb_poles_path)
    hyb_poles = vec(hyb_poles)
    hyb_weights_orig = readdlm(hyb_weights_path)
    size(hyb_weights_orig, 2) == HYB_WEIGHTS_EXPECTED_COLS || error("Hybridization weights file must have $(HYB_WEIGHTS_EXPECTED_COLS) columns (ind_poles, orb_1, orb_2, weights), got $(size(hyb_weights_orig, 2))")
    n_poles = Int(maximum(hyb_weights_orig[:, 1])) + 1
    n_flavors_physical = Int(maximum(hyb_weights_orig[:, 2:3])) + 1
    expected_rows = n_poles * n_flavors_physical * n_flavors_physical
    size(hyb_weights_orig, 1) == expected_rows || error("Hybridization weights file has incorrect number of rows: expected $expected_rows, got $(size(hyb_weights_orig, 1))")
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
    # Convert bare propagator e^{-tau H} to DLR representation
    bare_propagator_dlr = tau2dlr(dlr_grid, bare_propagator, dlr_grid.tau)
    bare_propagator_reconstructed = dlr2tau(dlr_grid, bare_propagator_dlr, dlr_grid.tau)
    bare_propagator_err = maximum(abs.(bare_propagator_reconstructed .- bare_propagator))


    # precompute annihilation and creation operators multiplied with SOE coefficients of the hybridization function
    ann_opts_SOE = _gen_opts_SOE(T, ann_opts, hyb_soe.coeffs, false)
    cre_opts_SOE = _gen_opts_SOE(T, cre_opts, hyb_soe.coeffs, true)

    # precompute K0_inverse
    K0_inverse = _gen_K0_inverse(hyb_poles, model_params.beta)

    # calculate partition function for the counter-terms
    Z_counterterms = calculate_Z_counterterms(model_params, n_flavors, n_flavors_physical)

    # Pre-compute kernel cache and workspace for DLR convolutions (performance optimization)
    kernel_cache = compute_kernel_cache(dlr_grid)
    r = length(dlr_grid.omega)
    n_tau = length(dlr_grid.tau)
    convolution_workspace = ConvolutionWorkspace{T}(r, n_fock, n_fock, n_tau)

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
        bare_propagator_err,
        hyb_soe,
        DLRData(bare_propagator, bare_propagator_dlr), # bare propagator
        K0_inverse,
        ann_opts_SOE,
        cre_opts_SOE,
        Z_counterterms,
        kernel_cache,
        convolution_workspace
    )
end

"""
    init_solver_data(model_data::ModelData)

init a `SolverData` object from the `ModelData`.
"""
function init_solver_data(model_data::ModelData{T}) where {T<:Number}

    calculation_data = CalculationData(
        DLRData{T}(model_data.dlr_grid.size, model_data.n_fock, model_data.n_fock), # pseudo-particle propagator
        DLRData{T}(model_data.dlr_grid.size, model_data.n_fock, model_data.n_fock), # pseudo-particle self-energy
        DLRData{T}(model_data.dlr_grid.size, model_data.n_flavors_physical, model_data.n_flavors_physical), # physical Green's function
        Float64[], # partition function by order (filled during eval_diagrams!)
        Vector{Array{T,3}}(), # green numerator by order (filled during eval_diagrams!)
        Vector{Array{T,3}}(), # green by order (computed at the end)
        zeros(T, 0, 0, 0), # green_linear_grid (filled during post-processing)
        Float64[] # tau_linear_grid (filled during post-processing)
    )
    return SolverData(model_data, calculation_data)
end

function _gen_K0_inverse(exponents::Vector{Float64}, beta::Float64)
    K0_inverse = zeros(Float64, length(exponents))
    for (i, omega) in enumerate(exponents)
        if omega <= 0
            K0_inverse[i] = 1.0 + exp(beta * omega)
        else
            K0_inverse[i] = 1.0 + exp(-beta * omega)
        end
    end
    # Validate: all values should be between 1 and 2
    (all(K0_inverse .>= 1.0) && all(K0_inverse .<= 2.0)) || error("K0_inverse values must be between 1 and 2, got range [$(minimum(K0_inverse)), $(maximum(K0_inverse))]")
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


function _gen_opts_SOE(::Type{T}, opts::Vector{Matrix{T}}, coeffs::Array{Float64,3}, is_creation::Bool) where {T<:Number}
    # coeffs shape: (p, o1, o2), o1 correspond to creation and o2 correspond to annihilation
    # \bar{F}^{\dagger}_{o, l} = \sum_{o'} Delta_{l, o', o} F^{\dagger}_{o'}
    # \bar{F}_{o, l} = \sum_{o'} Delta_{l, o, o'} F_{o'}

    n_flavors = length(opts)
    p = size(coeffs, 1)
    size(coeffs, 2) == n_flavors || error("Size mismatch between coeffs dimension 2 and number of flavors: expected $n_flavors, got $(size(coeffs, 2))")
    size(coeffs, 3) == n_flavors || error("Size mismatch between coeffs dimension 3 and number of flavors: expected $n_flavors, got $(size(coeffs, 3))")
    opts_SOE = Matrix{Matrix{T}}(undef, n_flavors, p)
    for o in 1:n_flavors
        for l in 1:p
            mat = zeros(T, size(opts[1])...)
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
