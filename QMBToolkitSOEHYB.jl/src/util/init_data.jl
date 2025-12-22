using QMBToolkitED
using QMBToolkitED.Dense: DenseOperator, DenseBasis
using QMBToolkitED.BlockDense
using DelimitedFiles
using QMBToolkitDLR
using LinearAlgebra

# Constants for file format validation
const HYB_WEIGHTS_EXPECTED_COLS = 4  # Expected columns: ind_poles, orb_1, orb_2, weights

#=============================================================================
# Main initialization functions
=============================================================================#

"""
    init_model_data(model_params::ModelParams, dlr_params::DLRParams)

Initialize a `ModelData{T, DenseFormat}` object from input parameters.
"""
function init_model_data(model_params::ModelParams, dlr_params::DLRParams)
    # Load and solve Hamiltonian
    hopping_path, interaction_path = _load_hamiltonian_paths(model_params)
    H, ann_opts_raw = QMBToolkitED.build_hamiltonian(hopping_path, interaction_path)
    evals, evecs = QMBToolkitED.solve_hamiltonian(H)

    # Infer numeric type and transform operators to eigenbasis
    T = eltype(evecs)
    n_flavors = length(ann_opts_raw)
    n_fock = 1 << n_flavors

    # Create eigenbasis for the transformed operators
    eigenbasis = DenseBasis(n_flavors)

    # Extract matrix data from DenseOperator and transform to eigenbasis, wrap in DenseOperator
    ann_opts = vec([DenseOperator(eigenbasis, evecs' * convert(Matrix{ComplexF64}, to_matrix(c)) * evecs) for c in ann_opts_raw])
    cre_opts = vec([DenseOperator(eigenbasis, Matrix{ComplexF64}(adjoint(c.data))) for c in ann_opts])

    # Initialize DLR grid
    dlr_grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)

    # Build bare propagator
    E_shift = minimum(evals)
    evals_shifted = evals .- E_shift
    bare_propagator = _build_bare_propagator_dense(T, dlr_grid.tau, evals_shifted, n_fock)

    # Load hybridization data
    hyb_soe, n_flavors_physical = _load_hybridization_data(model_params, n_flavors, dlr_grid)

    # Compute DLR quantities
    bare_propagator_dlr, bare_propagator_err, K0_inverse, kernel_cache, convolution_workspace =
        _compute_dlr_quantities(T, dlr_grid, bare_propagator, hyb_soe.exponents, model_params.beta, n_fock)

    # Precompute SOE operators
    ann_opts_SOE = _gen_opts_SOE(eigenbasis, ann_opts, hyb_soe.coeffs, false)
    cre_opts_SOE = _gen_opts_SOE(eigenbasis, cre_opts, hyb_soe.coeffs, true)

    # Calculate partition function for counter-terms
    Z_counterterms = calculate_Z_counterterms(model_params, n_flavors, n_flavors_physical)

    # Wrap bare propagator in DenseDLRPropagator
    bare_propagator_wrapped = DenseDLRPropagator(
        DensePropagator(bare_propagator),
        DensePropagator(bare_propagator_dlr)
    )

    # Create format-specific storage
    format_data = DenseModelDataStorage{T}(
        ann_opts,
        cre_opts,
        ann_opts_SOE,
        cre_opts_SOE,
        bare_propagator_wrapped
    )

    return ModelData{T, DenseFormat, DenseModelDataStorage{T}}(
        model_params.beta,
        n_flavors,
        n_fock,
        evals,
        evecs,
        dlr_grid,
        E_shift,
        n_flavors_physical,
        bare_propagator_err,
        hyb_soe,
        K0_inverse,
        Z_counterterms,
        kernel_cache,
        convolution_workspace,
        format_data
    )
end

"""
    init_model_data_blockdense(model_params::ModelParams, dlr_params::DLRParams)

Initialize a `ModelData{T, BlockDenseFormat}` object from input parameters.
"""
function init_model_data_blockdense(model_params::ModelParams, dlr_params::DLRParams)
    # Load Hamiltonian paths and infer basis
    hopping_path, interaction_path = _load_hamiltonian_paths(model_params)
    hopping_data = readdlm(hopping_path)
    n_orbitals = Int(maximum(hopping_data[:, 1:2])) + 1

    # Build basis assuming particle number conservation
    site_qns = [QN_N(1) for _ in 1:n_orbitals]
    basis = make_basis(site_qns)

    # Build and solve BlockDense Hamiltonian
    H_block, c_block = build_hamiltonian(hopping_path, interaction_path; method=:blockdense, basis=basis)
    result = solve_hamiltonian_blockdense(H_block)

    # Infer numeric type and extract operators
    T = eltype(result.sector_eigenvectors[1])
    ann_opts_blockdense = c_block
    cre_opts_blockdense = [adjoint(c) for c in c_block]
    n_flavors = length(ann_opts_blockdense)
    n_fock = sum(sector.dim for sector in basis.sectors)

    # Extract eigenvalues (concatenate from all sectors)
    evals = vcat([result.sector_eigenvalues[i] for i in 1:length(basis.sectors)]...)

    # Build block-diagonal eigenvector matrix
    evecs = zeros(T, n_fock, n_fock)
    offset = 0
    for i in 1:length(basis.sectors)
        dim = basis.sectors[i].dim
        evecs[offset+1:offset+dim, offset+1:offset+dim] = result.sector_eigenvectors[i]
        offset += dim
    end

    # Initialize DLR grid
    dlr_grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)

    # Build bare propagators (dense for DLR, BlockDense for computation)
    E_shift = minimum(evals)
    evals_shifted = evals .- E_shift
    bare_propagator_dense = _build_bare_propagator_dense(T, dlr_grid.tau, evals_shifted, n_fock)
    bare_propagator_blockdense_tau = build_bare_propagator_blockdense(result, dlr_grid.tau, E_shift)

    # Load hybridization data
    hyb_soe, n_flavors_physical = _load_hybridization_data(model_params, n_flavors, dlr_grid)

    # Compute DLR quantities
    bare_propagator_dlr, bare_propagator_err, K0_inverse, kernel_cache, convolution_workspace =
        _compute_dlr_quantities(T, dlr_grid, bare_propagator_dense, hyb_soe.exponents, model_params.beta, n_fock)

    # Create BlockDense SOE operators
    ann_opts_SOE_blockdense = _gen_opts_SOE_blockdense(T, ann_opts_blockdense, hyb_soe.coeffs, false)
    cre_opts_SOE_blockdense = _gen_opts_SOE_blockdense(T, cre_opts_blockdense, hyb_soe.coeffs, true)

    # Calculate partition function for counter-terms
    Z_counterterms = calculate_Z_counterterms(model_params, n_flavors, n_flavors_physical)

    # Compute DLR coefficients for BlockDense propagator
    bare_propagator_blockdense_dlr = tau2dlr_blockdense(dlr_grid, bare_propagator_blockdense_tau, dlr_grid.tau)
    bare_propagator_blockdense_wrapped = BlockDenseDLRPropagator(
        bare_propagator_blockdense_tau,
        bare_propagator_blockdense_dlr
    )

    # Create format-specific storage
    format_data = BlockDenseModelDataStorage{T}(
        ann_opts_blockdense,
        cre_opts_blockdense,
        ann_opts_SOE_blockdense,
        cre_opts_SOE_blockdense,
        bare_propagator_blockdense_wrapped
    )

    return ModelData{T, BlockDenseFormat, BlockDenseModelDataStorage{T}}(
        model_params.beta,
        n_flavors,
        n_fock,
        evals,
        evecs,
        dlr_grid,
        E_shift,
        n_flavors_physical,
        bare_propagator_err,
        hyb_soe,
        K0_inverse,
        Z_counterterms,
        kernel_cache,
        convolution_workspace,
        format_data
    )
end


"""
    init_solver_data(model_data::ModelData)

Initialize a `SolverData` object from the `ModelData`.
"""
function init_solver_data(model_data::ModelData{T, DenseFormat, DenseModelDataStorage{T}}) where {T<:Number}
    r = model_data.dlr_grid.size
    n_fock = model_data.n_fock
    n_phys = model_data.n_flavors_physical

    # Create format-specific storage
    format_data = DenseCalculationDataStorage{T}(
        DenseDLRPropagator{T}(r, n_fock, n_fock),  # propagator
        DenseDLRPropagator{T}(r, n_fock, n_fock)   # selfenergy
    )

    calculation_data = CalculationData{T, DenseFormat, DenseCalculationDataStorage{T}}(
        format_data,
        DenseDLRPropagator{T}(r, n_phys, n_phys),  # green
        Float64[],  # partition function by order
        Vector{DensePropagator{T}}(),  # green numerator by order
        Vector{DensePropagator{T}}(),  # green by order
        zeros(DensePropagator{T}, 0, n_phys, n_phys),  # green_linear_grid
        Float64[]  # tau_linear_grid
    )
    return SolverData(model_data, calculation_data)
end

function init_solver_data(model_data::ModelData{T, BlockDenseFormat, BlockDenseModelDataStorage{T}}) where {T<:Number}
    # Get basis and flux from bare propagator
    bare_prop = get_bare_propagator(model_data)
    basis = bare_prop.vals.basis
    flux = bare_prop.vals.flux
    r = model_data.dlr_grid.size
    n_phys = model_data.n_flavors_physical

    # Create format-specific storage
    format_data = BlockDenseCalculationDataStorage{T}(
        BlockDenseDLRPropagator{T}(basis, flux, r),  # propagator
        BlockDenseDLRPropagator{T}(basis, flux, r)   # selfenergy
    )

    calculation_data = CalculationData{T, BlockDenseFormat, BlockDenseCalculationDataStorage{T}}(
        format_data,
        DenseDLRPropagator{T}(r, n_phys, n_phys),  # green
        Float64[],  # partition function by order
        Vector{DensePropagator{T}}(),  # green numerator by order
        Vector{DensePropagator{T}}(),  # green by order
        zeros(DensePropagator{T}, 0, n_phys, n_phys),  # green_linear_grid
        Float64[]  # tau_linear_grid
    )
    return SolverData(model_data, calculation_data)
end

#=============================================================================
# Helper functions
=============================================================================#

"""
    _load_hamiltonian_paths(model_params::ModelParams)

Validate and return paths to Hamiltonian input files.
"""
function _load_hamiltonian_paths(model_params::ModelParams)
    hopping_path = joinpath(model_params.file_path, model_params.hopping_file)
    interaction_path = joinpath(model_params.file_path, model_params.interaction_file)
    isfile(hopping_path) || error("Hopping file not found: $hopping_path")
    isfile(interaction_path) || error("Interaction file not found: $interaction_path")
    return hopping_path, interaction_path
end

"""
    _load_hybridization_data(model_params::ModelParams, n_flavors::Int, dlr_grid::DLRGrid)

Load hybridization function data from files and construct SOEData.
Returns (hyb_soe, n_flavors_physical).
"""
function _load_hybridization_data(model_params::ModelParams, n_flavors::Int, dlr_grid::DLRGrid)
    hyb_poles_path = joinpath(model_params.file_path, model_params.hybridization_poles_file)
    hyb_weights_path = joinpath(model_params.file_path, model_params.hybridization_weights_file)
    isfile(hyb_poles_path) || error("Hybridization poles file not found: $hyb_poles_path")
    isfile(hyb_weights_path) || error("Hybridization weights file not found: $hyb_weights_path")

    hyb_poles = vec(readdlm(hyb_poles_path))
    hyb_weights_orig = readdlm(hyb_weights_path)
    size(hyb_weights_orig, 2) == HYB_WEIGHTS_EXPECTED_COLS || error("Hybridization weights file must have $(HYB_WEIGHTS_EXPECTED_COLS) columns (ind_poles, orb_1, orb_2, weights), got $(size(hyb_weights_orig, 2))")

    n_poles = Int(maximum(hyb_weights_orig[:, 1])) + 1
    n_flavors_physical = Int(maximum(hyb_weights_orig[:, 2:3])) + 1
    expected_rows = n_poles * n_flavors_physical * n_flavors_physical
    size(hyb_weights_orig, 1) == expected_rows || error("Hybridization weights file has incorrect number of rows: expected $expected_rows, got $(size(hyb_weights_orig, 1))")

    hyb_weights_flat = hyb_weights_orig[:, 4]
    hyb_weights_permuted = reshape(hyb_weights_flat, (n_flavors_physical, n_flavors_physical, n_poles))
    hyb_weights_physical = permutedims(hyb_weights_permuted, (3, 2, 1))

    # hyb_weights contains both physical weights and counterterms (zeros)
    hyb_weights = zeros(Float64, n_poles, n_flavors, n_flavors)
    hyb_weights[:, 1:n_flavors_physical, 1:n_flavors_physical] = hyb_weights_physical

    # Convert hybridization function to SOE representation
    KPlus_on_DLRtau = _gen_K_on_DLRtau(hyb_poles, dlr_grid)
    KMinus_on_DLRtau = _gen_K_on_DLRtau(-hyb_poles, dlr_grid)
    hyb_soe = SOEData(hyb_weights, hyb_poles, KPlus_on_DLRtau, KMinus_on_DLRtau)

    return hyb_soe, n_flavors_physical
end

"""
    _build_bare_propagator_dense(T, tau_grid, evals_shifted, n_fock)

Build dense bare propagator on tau grid (diagonal in eigenbasis).
"""
function _build_bare_propagator_dense(::Type{T}, tau_grid, evals_shifted, n_fock) where T
    bare_propagator = zeros(T, length(tau_grid), n_fock, n_fock)
    for (k, tau) in enumerate(tau_grid)
        U_tau = exp.(-tau * evals_shifted)
        bare_propagator[k, :, :] = Diagonal(U_tau)
    end
    return bare_propagator
end

"""
    _compute_dlr_quantities(T, dlr_grid, bare_propagator, hyb_poles, beta, n_fock)

Compute DLR-related quantities: bare_propagator_dlr, error, K0_inverse, caches.
"""
function _compute_dlr_quantities(::Type{T}, dlr_grid, bare_propagator, hyb_poles, beta, n_fock) where T
    bare_propagator_dlr = tau2dlr(dlr_grid, bare_propagator, dlr_grid.tau)
    bare_propagator_reconstructed = dlr2tau(dlr_grid, bare_propagator_dlr, dlr_grid.tau)
    bare_propagator_err = maximum(abs.(bare_propagator_reconstructed .- bare_propagator))

    K0_inverse = _gen_K0_inverse(hyb_poles, beta)
    kernel_cache = compute_kernel_cache(dlr_grid)
    r = length(dlr_grid.omega)
    n_tau = length(dlr_grid.tau)
    convolution_workspace = ConvolutionWorkspace{T}(r, n_fock, n_fock, n_tau)

    return bare_propagator_dlr, bare_propagator_err, K0_inverse, kernel_cache, convolution_workspace
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

function _gen_opts_SOE(eigenbasis::DenseBasis, opts::Vector{DenseOperator}, coeffs::Array{Float64,3}, is_creation::Bool)
    # coeffs shape: (p, o1, o2), o1 correspond to creation and o2 correspond to annihilation
    # \bar{F}^{\dagger}_{o, l} = \sum_{o'} Delta_{l, o', o} F^{\dagger}_{o'}
    # \bar{F}_{o, l} = \sum_{o'} Delta_{l, o, o'} F_{o'}

    n_flavors = length(opts)
    p = size(coeffs, 1)
    size(coeffs, 2) == n_flavors || error("Size mismatch between coeffs dimension 2 and number of flavors: expected $n_flavors, got $(size(coeffs, 2))")
    size(coeffs, 3) == n_flavors || error("Size mismatch between coeffs dimension 3 and number of flavors: expected $n_flavors, got $(size(coeffs, 3))")
    opts_SOE = Matrix{DenseOperator}(undef, n_flavors, p)
    for o in 1:n_flavors
        for l in 1:p
            mat = zeros(ComplexF64, size(opts[1].data)...)
            if is_creation
                for o_prime in 1:n_flavors
                    mat .+= coeffs[l, o_prime, o] * opts[o_prime].data
                end
            else
                for o_prime in 1:n_flavors
                    mat .+= coeffs[l, o, o_prime] * opts[o_prime].data
                end
            end
            opts_SOE[o, l] = DenseOperator(eigenbasis, mat)
        end
    end
    return opts_SOE
end

function _gen_opts_SOE_blockdense(::Type{T}, opts::Vector{BlockDenseOperator}, coeffs::Array{Float64,3}, is_creation::Bool) where {T<:Number}
    # BlockDense version: create linear combinations of BlockDenseOperators
    # coeffs shape: (p, o1, o2), o1 correspond to creation and o2 correspond to annihilation
    # \bar{F}^{\dagger}_{o, l} = \sum_{o'} Delta_{l, o', o} F^{\dagger}_{o'}
    # \bar{F}_{o, l} = \sum_{o'} Delta_{l, o, o'} F_{o'}

    n_flavors = length(opts)
    p = size(coeffs, 1)
    size(coeffs, 2) == n_flavors || error("Size mismatch between coeffs dimension 2 and number of flavors: expected $n_flavors, got $(size(coeffs, 2))")
    size(coeffs, 3) == n_flavors || error("Size mismatch between coeffs dimension 3 and number of flavors: expected $n_flavors, got $(size(coeffs, 3))")

    opts_SOE = Matrix{BlockDenseOperator}(undef, n_flavors, p)
    for o in 1:n_flavors
        for l in 1:p
            # Start with zero operator (same structure as opts[1])
            result = coeffs[l, 1, 1] * opts[1]  # Initialize with scaled first operator
            result = result - result  # Set to zero (preserves structure)

            if is_creation
                for o_prime in 1:n_flavors
                    result = result + coeffs[l, o_prime, o] * opts[o_prime]
                end
            else
                for o_prime in 1:n_flavors
                    result = result + coeffs[l, o, o_prime] * opts[o_prime]
                end
            end
            opts_SOE[o, l] = result
        end
    end
    return opts_SOE
end
