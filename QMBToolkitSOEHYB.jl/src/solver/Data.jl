using QMBToolkitDLR
using QMBToolkitED: Operator, Basis, DiagonalOperator, QuantumNumber

#=============================================================================
# SOE Data
=============================================================================#

"""
    SOEData

Sum-of-Exponentials representation of the hybridization function.
The value is recovered by sum_{i} coeffs[i] * K(tau, exponents[i]).
Different matrix elements share the same exponents.
"""
mutable struct SOEData
    coeffs::Array{Float64, 3}         # SOE coefficients, shape: (p, o1, o2)
    exponents::Vector{Float64}       # SOE exponents, shape: (p)
    KPlus_on_DLRtau::Array{Float64, 2}  # K(tau_k, exponent_i) on the DLR tau grid
    KMinus_on_DLRtau::Array{Float64, 2} # K(tau_k, -exponent_i) on the DLR tau grid
end

#=============================================================================
# Format-Specific Storage Types for ModelData
=============================================================================#

"""
    ModelDataStorage{T}

Storage for model data holding eigenvalues, eigenvectors, operators,
and bare propagator.
"""
struct ModelDataStorage{T}
    eigenvalues::DiagonalOperator     # Eigenvalues as block-diagonal operator
    eigenvectors::Operator            # Eigenvectors as block-unitary operator
    ann_opts::Vector{Operator}        # Annihilation operators (eigenbasis)
    cre_opts::Vector{Operator}        # Creation operators (eigenbasis)
    ann_opts_bar::Matrix{Operator}    # Barred annihilation operators c̄_{o,l}
    cre_opts_bar::Matrix{Operator}    # Barred creation operators c̄†_{o,l}
    bare_propagator::DLRPropagator{T}           # Bare propagator in DLR representation
end

#=============================================================================
# Format-Specific Storage Types for CalculationData
=============================================================================#

"""
    CalculationDataStorage{T}

Storage for calculation data holding propagator and self-energy.
"""
struct CalculationDataStorage{T}
    propagator::DLRPropagator{T}  # Propagator in DLR representation
    selfenergy::DLRPropagator{T}  # Self-energy in DLR representation
end

#=============================================================================
# ModelData
=============================================================================#

"""
    ModelData{T}

Holds the derived data for the model, such as the Hamiltonian, operators, and hybridization function.

The `format_data` field contains operators and bare propagator.

# Type Parameters

  - `T<:Number`: Numeric type for propagator values (typically ComplexF64)
"""
struct ModelData{T <: Number}
    # Model properties
    beta::Float64
    n_flavors::Int
    n_flavors_physical::Int
    n_fock::Int
    E_shift::Float64
    Z_counterterms::Float64
    bare_propagator_err::Float64  # Diagnostic: max DLR reconstruction error

    # Hybridization data
    hyb::SOEData
    K0_inverse::Vector{Float64}

    # Numerical infrastructure
    dlr_grid::DLRGrid{Float64, :none}
    kernel_cache::KernelCache
    convolution_workspace::ConvolutionWorkspace{T}

    # Format-specific storage (eigenvalues, eigenvectors, operators, bare propagator)
    format_data::ModelDataStorage{T}
end

#=============================================================================
# CalculationData
=============================================================================#

"""
    CalculationData{T}

Holds the data for the calculation, such as the Green's function, propagator, etc.

# Type Parameters

  - `T<:Number`: Numeric type for propagator values
"""
mutable struct CalculationData{T <: Number}
    # Format-specific storage
    format_data::CalculationDataStorage{T}

    # Green's function (physical observable in orbital space)
    green::GreenDLR{T}
    partition_function_by_order::Vector{Float64}
    green_numerator_by_order::Vector{Array{T, 3}}
    green_by_order::Vector{Array{T, 3}}
    green_linear_grid::Array{T, 3}
    tau_linear_grid::Vector{Float64}

    # Diagram count information: (n_topologies, n_directions, n_flavors_combo, n_exponents_combo, total)
    # Index corresponds to expansion order (propagator: 0-based stored at index order+1, green: 1-based)
    propagator_diagram_info::Vector{NTuple{5, Int}}
    green_diagram_info::Vector{NTuple{5, Int}}
end

#=============================================================================
# SolverData
=============================================================================#

"""
    SolverData{T}

Holds all the data required for the solver calculation.
Combines model data and calculation data.

# Type Parameters

  - `T<:Number`: Numeric type for propagator values
"""
struct SolverData{T <: Number}
    model::ModelData{T}
    calc::CalculationData{T}
end

#=============================================================================
# Accessor Functions (Unified Interface)
#
# These provide a format-independent way to access format-specific data.
# Simple accessors like these are typically inlined by Julia's compiler (zero overhead).
# For clarity and guaranteed performance, cache the result in a local variable
# when using it multiple times within the same function.
=============================================================================#

# ModelData accessors
get_ann_opts(model::ModelData) = model.format_data.ann_opts
get_cre_opts(model::ModelData) = model.format_data.cre_opts
get_ann_opts_bar(model::ModelData) = model.format_data.ann_opts_bar
get_cre_opts_bar(model::ModelData) = model.format_data.cre_opts_bar
get_bare_propagator(model::ModelData) = model.format_data.bare_propagator

# CalculationData accessors
get_propagator(calc::CalculationData) = calc.format_data.propagator
get_selfenergy(calc::CalculationData) = calc.format_data.selfenergy
