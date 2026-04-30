using QMBToolkitED: Basis, Operator, DiagonalOperator,
                    QN_N, QN_NSz, QuantumNumber,
                    build_hamiltonian, solve_hamiltonian
using DelimitedFiles
using QMBToolkitDLR
using LinearAlgebra

# Constants for file format validation
const HYB_WEIGHTS_EXPECTED_COLS = 4  # Expected columns: ind_poles, orb_1, orb_2, weights

#=============================================================================
# Spin conservation helpers
=============================================================================#

"""
    _get_spin(flavor::Int) -> Int

Extract spin index from flavor index (0-indexed).
Convention: flavor = 2*orbital + spin, where spin ∈ {0=up, 1=down}.
Returns 0 for spin-up, 1 for spin-down.
"""
_get_spin(flavor::Int) = flavor % 2

"""
    _check_spin_conservation(hopping_path::String, interaction_path::String) -> Bool

Check if the Hamiltonian conserves spin based on hopping and Coulomb interaction terms.

Returns `true` if spin is conserved (all terms satisfy spin selection rules).
Returns `false` if:

  - Number of flavors is odd (no spin structure possible with 2 spins per orbital)
  - Any hopping term violates spin conservation (t[i,j] != 0 with spin(i) != spin(j))
  - Any Coulomb term violates spin conservation (U[i,j,k,l] != 0 with spin(i)+spin(j) != spin(k)+spin(l))

When spin conservation is violated, a warning is logged.
"""
function _check_spin_conservation(hopping_path::String, interaction_path::String)
    # Step 1: Read hopping file and determine n_flavors
    hopping_data = readdlm(hopping_path)
    n_flavors = Int(maximum(hopping_data[:, 1:2])) + 1

    # Step 2: Check for odd number of flavors (no spin structure with 2 spins per orbital)
    if n_flavors % 2 != 0
        @warn "Odd number of flavors ($n_flavors): spin conservation not applicable, using QN_N"
        return false
    end

    # Step 3: Check hopping terms for spin conservation
    # Hopping conserves spin if: t[i,j] != 0 => spin(i) == spin(j)
    for row_idx in 1:size(hopping_data, 1)
        i = Int(hopping_data[row_idx, 1])  # 0-indexed flavor
        j = Int(hopping_data[row_idx, 2])  # 0-indexed flavor
        re = hopping_data[row_idx, 3]
        im = hopping_data[row_idx, 4]

        # Skip zero entries
        if re == 0.0 && im == 0.0
            continue
        end

        # Check spin conservation
        if _get_spin(i) != _get_spin(j)
            @warn "Spin conservation violated in hopping: t[$i,$j] != 0 but spins differ (spin($i)=$(_get_spin(i)) vs spin($j)=$(_get_spin(j))), using QN_N"
            return false
        end
    end

    # Step 4: Read and check interaction file
    interaction_lines = readlines(interaction_path)

    # First line is the count of interaction terms
    if isempty(interaction_lines)
        # No interaction file content - spin is conserved by default
        return true
    end

    n_terms = parse(Int, strip(interaction_lines[1]))
    if n_terms == 0
        # No interaction terms - spin is conserved
        return true
    end

    # Check each Coulomb term: U[i,j,k,l] c†_i c†_j c_k c_l
    # Spin conservation: spin(i) + spin(j) == spin(k) + spin(l)
    for line_idx in 2:length(interaction_lines)
        line = strip(interaction_lines[line_idx])
        if isempty(line)
            continue
        end

        parts = split(line)
        # Format: idx i j k l Re(U) Im(U)
        fl_i = parse(Int, parts[2])  # 0-indexed
        fl_j = parse(Int, parts[3])  # 0-indexed
        fl_k = parse(Int, parts[4])  # 0-indexed
        fl_l = parse(Int, parts[5])  # 0-indexed
        re = parse(Float64, parts[6])
        im = parse(Float64, parts[7])

        # Skip zero entries
        if re == 0.0 && im == 0.0
            continue
        end

        # Check spin conservation for chemist notation: U[i,j,k,l] c†_i c†_k c_l c_j
        # Creation operators: i, k; Annihilation operators: l, j
        spin_in = _get_spin(fl_i) + _get_spin(fl_k)
        spin_out = _get_spin(fl_l) + _get_spin(fl_j)

        if spin_in != spin_out
            @warn "Spin conservation violated in Coulomb: U[$fl_i,$fl_j,$fl_k,$fl_l] != 0 but spin change is $(spin_in - spin_out), using QN_N"
            return false
        end
    end

    # All checks passed - spin is conserved
    return true
end

#=============================================================================
# Main initialization function
=============================================================================#

"""
    init_model_data(model_params::ModelParams, solver_params::SolverParams, rank::Int)

Initialize a `ModelData` object from input parameters.

# Arguments

  - `model_params::ModelParams`: Model parameters from TOML file
  - `solver_params::SolverParams`: Solver parameters from TOML file
  - `rank::Int`: MPI rank (for controlling print output)

# Returns

  - `ModelData{T}`: Initialized model data
"""
function init_model_data(model_params::ModelParams, solver_params::SolverParams, rank::Int)
    dlr_params = solver_params.dlr
    is_root = (rank == 0)

    # Load Hamiltonian paths and build basis
    hopping_path, interaction_path = _load_hamiltonian_paths(model_params)
    basis = _build_basis(hopping_path, interaction_path)

    # Build and solve Hamiltonian
    H, ann_opts_raw = build_hamiltonian(basis, hopping_path, interaction_path)
    result = solve_hamiltonian(H)
    is_root && println("Exact diagonalization complete")

    eigenvalues = result.eigenvalues
    eigenvectors = result.eigenvectors

    # Extract system information
    T, n_flavors, n_fock = _get_system_info(eigenvectors, ann_opts_raw)

    # Initialize DLR grid
    dlr_grid = DLRGrid(dlr_params.Euv, model_params.beta, dlr_params.rtol, true)

    # Build bare propagator (diagonal in eigenbasis)
    E_shift = _get_E_shift(eigenvalues)
    bare_propagator_tau = _build_bare_propagator(T, eigenvalues, dlr_grid.τ, E_shift,
        model_params.beta, solver_params.prune_threshold, is_root)

    # Transform operators to eigenbasis
    ann_opts = _transform_to_eigenbasis(ann_opts_raw, eigenvectors)
    cre_opts = [adjoint(c) for c in ann_opts]
    is_root && println("Eigenbasis transformation complete")

    # Load hybridization data
    hyb_soe, n_flavors_physical = _load_hybridization_data(model_params, n_flavors, dlr_grid)

    # Convert bare propagator to DLR representation and compute error
    bare_propagator_dlr = _convert_to_dlr(dlr_grid, bare_propagator_tau)
    bare_propagator_err = _compute_dlr_error(dlr_grid, bare_propagator_tau, bare_propagator_dlr)

    # Compute hybridization kernel normalization factors
    K0_inverse = _gen_K0_inverse(hyb_soe.exponents, model_params.beta)

    # Precompute DLR kernel cache and convolution workspace
    kernel_cache = compute_kernel_cache(dlr_grid)
    convolution_workspace = _create_convolution_workspace(T, dlr_grid, bare_propagator_tau)

    # Precompute barred operators (only for physical flavors; counterterm flavors
    # have zero hybridization weight, so their barred operators are never used)
    ann_opts_bar = _gen_opts_bar(ann_opts, hyb_soe.coeffs, :annihilation, n_flavors_physical)
    cre_opts_bar = _gen_opts_bar(cre_opts, hyb_soe.coeffs, :creation, n_flavors_physical)
    is_root && println("Barred operators complete")

    # Calculate partition function for counter-terms
    Z_counterterms = _calculate_Z_counterterms(model_params, n_flavors, n_flavors_physical)

    # Wrap bare propagator
    bare_propagator_wrapped = _wrap_bare_propagator(bare_propagator_tau, bare_propagator_dlr)

    # Create format-specific storage
    format_data = ModelDataStorage{T}(
        eigenvalues,
        eigenvectors,
        ann_opts,
        cre_opts,
        ann_opts_bar,
        cre_opts_bar,
        bare_propagator_wrapped
    )

    return ModelData{T}(
        # Model properties
        model_params.beta,
        n_flavors,
        n_flavors_physical,
        n_fock,
        E_shift,
        Z_counterterms,
        bare_propagator_err,
        # Hybridization data
        hyb_soe,
        K0_inverse,
        # Numerical infrastructure
        dlr_grid,
        kernel_cache,
        convolution_workspace,
        # Format-specific storage
        format_data
    )
end

"""
    init_solver_data(model_data::ModelData)

Initialize a `SolverData` object from the `ModelData`.
"""
function init_solver_data(model_data::ModelData{T}) where {T <: Number}
    r = model_data.dlr_grid.size
    n_phys = model_data.n_flavors_physical

    # Create calculation storage
    bare_prop = get_bare_propagator(model_data)
    basis = bare_prop.vals.basis
    flux = bare_prop.vals.flux
    format_data = CalculationDataStorage{T}(
        DLRPropagator{T}(basis, flux, r),  # propagator
        DLRPropagator{T}(basis, flux, r)   # selfenergy
    )

    # Create CalculationData with common fields
    calculation_data = CalculationData{T}(
        format_data,
        GreenDLR{T}(r, n_phys, n_phys),  # green
        Float64[],  # partition function by order
        Vector{Array{T, 3}}(),  # green numerator by order
        Vector{Array{T, 3}}(),  # green by order
        zeros(T, 0, n_phys, n_phys),  # green_linear_grid
        Float64[],  # tau_linear_grid
        NTuple{5, Int}[],  # propagator_diagram_info
        NTuple{5, Int}[]   # green_diagram_info
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
    _build_basis(hopping_path::String, interaction_path::String)

Build a Basis from the hopping and interaction files.

If the Hamiltonian conserves spin (and n_flavors is even), uses QN_NSz quantum numbers
for more efficient block structure. Otherwise falls back to QN_N (particle number only).
"""
function _build_basis(hopping_path::String, interaction_path::String)
    hopping_data = readdlm(hopping_path)
    n_flavors = Int(maximum(hopping_data[:, 1:2])) + 1

    if _check_spin_conservation(hopping_path, interaction_path)
        # Spin conserved: use QN_NSz
        # Convention: flavor f = 2*orbital + spin, spin=0->sz=+1, spin=1->sz=-1
        qns = [QN_NSz(1, _get_spin(f) == 0 ? 1 : -1) for f in 0:(n_flavors - 1)]
    else
        # Spin not conserved: use QN_N only
        qns = [QN_N(1) for _ in 1:n_flavors]
    end

    return Basis(qns)
end

"""
    _get_system_info(eigenvectors::Operator{T}, ann_opts_raw::Vector{Operator{S}})

Extract system information (T, n_flavors, n_fock) from eigenvectors and operators.
Returns (T, n_flavors, n_fock) where T is the element type of eigenvectors.
"""
function _get_system_info(eigenvectors::Operator{T}, ann_opts_raw::Vector{<:Operator}) where {T}
    n_flavors = length(ann_opts_raw)
    n_fock = sum(sector.dim for sector in eigenvectors.basis.sectors)
    return T, n_flavors, n_fock
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
    size(hyb_weights_orig, 2) == HYB_WEIGHTS_EXPECTED_COLS ||
        error("Hybridization weights file must have $(HYB_WEIGHTS_EXPECTED_COLS) columns (ind_poles, orb_1, orb_2, weights), got $(size(hyb_weights_orig, 2))")

    n_poles = Int(maximum(hyb_weights_orig[:, 1])) + 1
    n_flavors_physical = Int(maximum(hyb_weights_orig[:, 2:3])) + 1
    expected_rows = n_poles * n_flavors_physical * n_flavors_physical
    size(hyb_weights_orig, 1) == expected_rows ||
        error("Hybridization weights file has incorrect number of rows: expected $expected_rows, got $(size(hyb_weights_orig, 1))")

    hyb_weights_flat = hyb_weights_orig[:, 4]
    hyb_weights_permuted = reshape(hyb_weights_flat, (
        n_flavors_physical, n_flavors_physical, n_poles))
    hyb_weights_physical = permutedims(hyb_weights_permuted, (3, 2, 1))

    # hyb_weights contains both weights for physical orbitals and those for counterterms (which are trivially zeros)
    hyb_weights = zeros(Float64, n_poles, n_flavors, n_flavors)
    hyb_weights[:, 1:n_flavors_physical, 1:n_flavors_physical] = hyb_weights_physical

    # Convert hybridization function to SOE representation
    KPlus_on_DLRtau = _gen_K_on_DLRtau(hyb_poles, dlr_grid)
    KMinus_on_DLRtau = _gen_K_on_DLRtau(-hyb_poles, dlr_grid)
    hyb_soe = SOEData(hyb_weights, hyb_poles, KPlus_on_DLRtau, KMinus_on_DLRtau)

    return hyb_soe, n_flavors_physical
end

#=============================================================================
# Bare propagator construction
=============================================================================#

"""
    _get_E_shift(eigenvalues::DiagonalOperator)

Get the energy shift (minimum eigenvalue) from a DiagonalOperator.
"""
function _get_E_shift(eigenvalues::DiagonalOperator)
    minimum(minimum(sector_evals) for (_, sector_evals) in eigenvalues.sector_diags)
end

"""
    _build_bare_propagator(T, eigenvalues, tau_grid, E_shift, beta, prune_threshold, is_root)

Build bare propagator on tau grid from eigenvalues.

The propagator is diagonal in the eigenbasis:
G[sector](%CF%84) = diag(exp(-τ * (E - E_shift)))

No eigenvector transformation is needed since we work in the eigenbasis.

If `prune_threshold > 0`, sectors where max(exp(-β * (E - E_shift))) < prune_threshold
are pruned (not included in the propagator).
"""
function _build_bare_propagator(::Type{T}, eigenvalues::DiagonalOperator,
        tau_grid::Vector{Float64}, E_shift::Float64,
        beta::Float64, prune_threshold::Float64, is_root::Bool) where {T}
    basis = eigenvalues.basis
    n_tau = length(tau_grid)

    # The propagator has flux = 0 (diagonal in sector space)
    flux = zero(typeof(basis.sectors[1].qn))

    blocks = Dict{Tuple{Int, Int}, Array{T, 3}}()

    n_total = length(basis.sectors)
    n_pruned = 0

    if is_root && prune_threshold > 0
        println("Bare propagator sector pruning (threshold = $prune_threshold):")
    end

    # Build diagonal blocks only (propagator conserves quantum numbers)
    for (sector_idx, sector) in enumerate(basis.sectors)
        sector_evals = eigenvalues.sector_diags[sector_idx]

        # Shift eigenvalues
        evals_shifted = sector_evals .- E_shift

        # Check if sector should be pruned: max value at τ=β is exp(-β * min_E)
        min_eval_shifted = minimum(evals_shifted)
        max_val_at_beta = exp(-beta * min_eval_shifted)

        if prune_threshold > 0 && max_val_at_beta < prune_threshold
            n_pruned += 1
            if is_root
                println("  Pruned sector $sector_idx: QN = $(sector.qn), dim = $(sector.dim), max_val_at_β = $max_val_at_beta")
            end
            continue
        end

        # Construct propagator for this sector (diagonal in eigenbasis)
        # Layout: (dim1, dim2, n_tau) for contiguous BLAS access
        sector_dim = sector.dim
        block = zeros(T, sector_dim, sector_dim, n_tau)

        for (k, tau) in enumerate(tau_grid)
            # G(τ) = diag(exp(-τ * (E - E_shift))) in eigenbasis
            U_tau = exp.(-tau .* evals_shifted)
            block[:, :, k] = Diagonal(U_tau)
        end

        blocks[(sector_idx, sector_idx)] = block
    end

    if is_root
        println("Bare propagator: kept $(n_total - n_pruned) / $n_total sectors" *
                (n_pruned > 0 ? " (pruned $n_pruned)" : ""))
    end

    return Propagator{T}(basis, flux, blocks)
end

#=============================================================================
# DLR quantities computation
=============================================================================#

"""
    _convert_to_dlr(dlr_grid, bare_propagator::Propagator)

Convert bare propagator from tau to DLR representation.
Returns the DLR coefficients as Propagator.
"""
function _convert_to_dlr(dlr_grid, bare_propagator::Propagator)
    return tau2dlr_propagator(dlr_grid, bare_propagator, dlr_grid.τ)
end

"""
    _compute_dlr_error(dlr_grid, bare_propagator_tau::Propagator, bare_propagator_dlr::Propagator)

Compute the maximum DLR reconstruction error.
Performs round-trip: tau -> dlr -> tau, then measures max absolute difference across all blocks.
"""
function _compute_dlr_error(dlr_grid, bare_propagator_tau::Propagator,
        bare_propagator_dlr::Propagator)
    bare_propagator_reconstructed = dlr2tau_propagator(dlr_grid, bare_propagator_dlr, dlr_grid.τ)

    max_err = 0.0
    for (block_key, block_tau) in bare_propagator_tau.blocks
        block_reconstructed = bare_propagator_reconstructed.blocks[block_key]
        block_err = maximum(abs.(block_reconstructed .- block_tau))
        max_err = max(max_err, block_err)
    end
    return max_err
end

"""
    _create_convolution_workspace(T, dlr_grid, prop::Propagator)

Create ConvolutionWorkspace sized for maximum sector dimension.
This workspace is reused for all block convolutions using `do_view=true` mode.
"""
function _create_convolution_workspace(::Type{T}, dlr_grid, prop::Propagator) where {T}
    # Size workspace for maximum sector dimension
    max_dim = maximum(sector.dim for sector in prop.basis.sectors)
    return ConvolutionWorkspace{T}(dlr_grid, max_dim, max_dim, max_dim)
end

"""
    _gen_K0_inverse(exponents::Vector{Float64}, beta::Float64)

Compute the inverse of K(τ=0, ω) for each SOE exponent.

For the kernel K(τ, ω) = exp(-ωτ) / (1 + exp(-βω)):

  - K(0, ω) = 1 / (1 + exp(-βω)) for ω > 0
  - K(0, ω) = 1 / (1 + exp(βω))  for ω ≤ 0

Returns vector of K(0, ω)^{-1} values, all in range [1, 2].
"""
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
    (all(K0_inverse .>= 1.0) && all(K0_inverse .<= 2.0)) ||
        error("K0_inverse values must be between 1 and 2, got range [$(minimum(K0_inverse)), $(maximum(K0_inverse))]")
    return K0_inverse
end

function _gen_K_on_DLRtau(exponents::Vector{Float64}, dlr_grid::DLRGrid)
    tau_grid = dlr_grid.τ
    K_on_DLRtau = zeros(Float64, length(tau_grid), length(exponents))
    for (j, omega) in enumerate(exponents)
        for (i, tau) in enumerate(tau_grid)
            K_on_DLRtau[i, j] = kernel_tau(tau, omega, dlr_grid.β)
        end
    end
    return K_on_DLRtau
end

#=============================================================================
# Propagator wrapping helpers
=============================================================================#

"""
    _wrap_bare_propagator(bare_propagator_tau::Propagator{T}, bare_propagator_dlr::Propagator{T}) where T

Wrap bare propagators into DLRPropagator.
"""
function _wrap_bare_propagator(bare_propagator_tau::Propagator{T},
        bare_propagator_dlr::Propagator{T}) where {T}
    return DLRPropagator(bare_propagator_tau, bare_propagator_dlr)
end

#=============================================================================
# Barred operator construction
=============================================================================#

"""
    _gen_opts_bar(opts::Vector{<:Operator}, coeffs::Array{Float64,3}, opt_type::Symbol,
                  n_flavors_physical::Int)

Generate barred operators for the physical flavor sector only.

coeffs shape: (p, o1, o2), o1 correspond to creation and o2 correspond to annihilation

  - For creation: c̄†_{o,l} = Σ_{o'} Δ_{l,o',o} c†_{o'}
  - For annihilation: c̄_{o,l} = Σ_{o'} Δ_{l,o,o'} c_{o'}

The outer index `o` and inner sum `o'` both run over physical flavors only.
Counterterm flavors carry zero hybridization weight, so their barred operators
are never accessed during diagram evaluation.

# Arguments

  - `opts`: Vector of operators (creation or annihilation), length == total n_flavors
  - `coeffs`: SOE coefficients array, shape (p, n_flavors, n_flavors)
  - `opt_type`: Either `:creation` or `:annihilation`
  - `n_flavors_physical`: Number of physical flavors; output is sized (n_flavors_physical, p)
"""
function _gen_opts_bar(opts::Vector{<:Operator}, coeffs::Array{Float64, 3}, opt_type::Symbol,
        n_flavors_physical::Int)
    opt_type in (:creation, :annihilation) ||
        error("opt_type must be :creation or :annihilation, got $opt_type")
    n_flavors = length(opts)
    p = size(coeffs, 1)
    size(coeffs, 2) == n_flavors ||
        error("Size mismatch between coeffs dimension 2 and number of flavors: expected $n_flavors, got $(size(coeffs, 2))")
    size(coeffs, 3) == n_flavors ||
        error("Size mismatch between coeffs dimension 3 and number of flavors: expected $n_flavors, got $(size(coeffs, 3))")
    n_flavors_physical <= n_flavors ||
        error("n_flavors_physical ($n_flavors_physical) must not exceed n_flavors ($n_flavors)")

    opts_bar = Matrix{Operator}(undef, n_flavors_physical, p)
    for o in 1:n_flavors_physical
        for l in 1:p
            # Find first non-zero coefficient to initialize result
            # This avoids adding operators with incompatible occupancy ranges
            first_nonzero_idx = 0
            if opt_type == :creation
                for o_prime in 1:n_flavors_physical
                    if coeffs[l, o_prime, o] != 0.0
                        first_nonzero_idx = o_prime
                        break
                    end
                end
            else
                for o_prime in 1:n_flavors_physical
                    if coeffs[l, o, o_prime] != 0.0
                        first_nonzero_idx = o_prime
                        break
                    end
                end
            end

            if first_nonzero_idx == 0
                # All coefficients are zero for this (o, l); produce a zero operator
                # with the correct block structure (preserves flux invariants downstream)
                result = 0.0 * opts[o]
            else
                # Initialize with the first non-zero term
                if opt_type == :creation
                    result = coeffs[l, first_nonzero_idx, o] * opts[first_nonzero_idx]
                else
                    result = coeffs[l, o, first_nonzero_idx] * opts[first_nonzero_idx]
                end

                # Add remaining non-zero terms
                if opt_type == :creation
                    for o_prime in (first_nonzero_idx + 1):n_flavors_physical
                        if coeffs[l, o_prime, o] != 0.0
                            result = result + coeffs[l, o_prime, o] * opts[o_prime]
                        end
                    end
                else
                    for o_prime in (first_nonzero_idx + 1):n_flavors_physical
                        if coeffs[l, o, o_prime] != 0.0
                            result = result + coeffs[l, o, o_prime] * opts[o_prime]
                        end
                    end
                end
            end
            opts_bar[o, l] = result
        end
    end
    return opts_bar
end

#=============================================================================
# Eigenbasis transformation
=============================================================================#

"""
    _transform_to_eigenbasis(ops::Vector{<:Operator}, evecs::Operator{T})

Transform a vector of operators to the eigenbasis.

For each operator, performs U' * op * U block-by-block, where U contains the eigenvectors.
This is analogous to the dense transformation: evecs' * op * evecs.

# Arguments

  - `ops::Vector{<:Operator}`: Operators in the original basis
  - `evecs::Operator{T}`: Eigenvector operator (block-diagonal, each block is unitary)

# Returns

  - `Vector{Operator{T}}`: Operators transformed to eigenbasis
"""
function _transform_to_eigenbasis(ops::Vector{<:Operator}, evecs::Operator{T}) where {T}
    transformed_ops = Vector{Operator{T}}(undef, length(ops))

    for (i, op) in enumerate(ops)
        # Transform each block: for block (tgt, src), we need U_tgt' * block * U_src
        new_blocks = Dict{Tuple{Int, Int}, Matrix{T}}()

        for ((tgt, src), block) in op.blocks
            # Get eigenvector matrices for source and target sectors
            U_tgt = evecs.blocks[(tgt, tgt)]
            U_src = evecs.blocks[(src, src)]

            # Transform: U_tgt' * block * U_src
            new_blocks[(tgt, src)] = adjoint(U_tgt) * block * U_src
        end

        transformed_ops[i] = Operator{T}(op.basis, op.flux, new_blocks)
    end

    return transformed_ops
end

#=============================================================================
# Counter-term partition function
=============================================================================#

"""
    _calculate_Z_counterterms(model_params::ModelParams, n_flavors::Int, n_flavors_physical::Int)

Calculate the partition function contribution from counter-term (unphysical) orbitals.

When n_flavors > n_flavors_physical, the extra orbitals are counter-terms with diagonal
hopping (on-site energies). Their partition function contribution is:
Z = ∏_i (1 + exp(-β * E_i))

Returns 1.0 if there are no counter-term orbitals.
"""
function _calculate_Z_counterterms(model_params::ModelParams, n_flavors::Int, n_flavors_physical::Int)
    if n_flavors == n_flavors_physical
        return 1.0
    end

    # Read hopping file in 'i j re im' format
    hopping_data = readdlm(joinpath(model_params.file_path, model_params.hopping_file))

    unphysical_dim = n_flavors - n_flavors_physical
    unphysical_hopping = zeros(ComplexF64, unphysical_dim, unphysical_dim)

    # Populate unphysical_hopping matrix
    # Assuming 0-indexed in file for i and j
    for row in eachrow(hopping_data)
        i, j = Int(row[1]), Int(row[2]) # 0-indexed

        # Check if both indices are in the unphysical range
        if i >= n_flavors_physical && j >= n_flavors_physical
            re, im = row[3], row[4]
            # Convert to 1-based index for our matrix
            row_idx = i - n_flavors_physical + 1
            col_idx = j - n_flavors_physical + 1
            unphysical_hopping[row_idx, col_idx] = re + im * 1im
        end
    end

    # Verify it's diagonal
    isdiag(unphysical_hopping) ||
        error("Unphysical part of hopping matrix must be diagonal, but found off-diagonal elements")

    # The diagonal elements are the energies of the unphysical sites
    unphysical_energies = real(diag(unphysical_hopping))

    # Partition function Z = product_i (1 + exp(-beta * E_i)) for fermions
    if isempty(unphysical_energies)
        return 1.0
    end
    Z_counterterms = prod(1 .+ exp.(-model_params.beta .* unphysical_energies))
    return Z_counterterms
end
