using QMBToolkitDLR
using QMBToolkitED.Dense: DenseOperator, DenseBasis, DenseDiagonalOperator
using QMBToolkitED.BlockDense: BlockDenseOperator, BlockDenseBasis, BlockDenseDiagonalOperator, OperatorFlux

#=============================================================================
# SOE Data (shared by both formats)
=============================================================================#

"""
    SOEData

Sum-of-Exponentials representation of the hybridization function.
The value is recovered by sum_{i} coeffs[i] * K(tau, exponents[i]).
Different matrix elements share the same exponents.
"""
mutable struct SOEData
    coeffs::Array{Float64,3}         # SOE coefficients, shape: (p, o1, o2)
    exponents::Vector{Float64}       # SOE exponents, shape: (p,)
    KPlus_on_DLRtau::Array{Float64,2}  # K(tau_k, exponent_i) on the DLR tau grid
    KMinus_on_DLRtau::Array{Float64,2} # K(tau_k, -exponent_i) on the DLR tau grid
end

#=============================================================================
# Format-Specific Storage Types for ModelData
=============================================================================#

"""
    ModelDataStorage{T}

Abstract type for format-specific model data storage.
Concrete subtypes hold operators and bare propagator in either Dense or BlockDense format.
"""
abstract type ModelDataStorage{T} end

"""
    DenseModelDataStorage{T}

Format-specific storage for DenseFormat model data.
Holds eigenvalues, eigenvectors, operators, and bare propagator in dense matrix representation.
"""
struct DenseModelDataStorage{T} <: ModelDataStorage{T}
    eigenvalues::DenseDiagonalOperator     # Eigenvalues as diagonal operator
    eigenvectors::DenseOperator            # Eigenvectors as unitary operator
    ann_opts::Vector{DenseOperator}        # Annihilation operators (eigenbasis)
    cre_opts::Vector{DenseOperator}        # Creation operators (eigenbasis)
    ann_opts_bar::Matrix{DenseOperator}    # Barred annihilation operators F̄_{o,l}
    cre_opts_bar::Matrix{DenseOperator}    # Barred creation operators F̄†_{o,l}
    bare_propagator::DenseDLRPropagator{T} # Bare propagator in DLR representation
end

"""
    BlockDenseModelDataStorage{T}

Format-specific storage for BlockDenseFormat model data.
Holds eigenvalues, eigenvectors, operators, and bare propagator in block-sparse representation.
"""
struct BlockDenseModelDataStorage{T} <: ModelDataStorage{T}
    eigenvalues::BlockDenseDiagonalOperator     # Eigenvalues as block-diagonal operator
    eigenvectors::BlockDenseOperator            # Eigenvectors as block-unitary operator
    ann_opts::Vector{BlockDenseOperator}        # Annihilation operators (eigenbasis)
    cre_opts::Vector{BlockDenseOperator}        # Creation operators (eigenbasis)
    ann_opts_bar::Matrix{BlockDenseOperator}    # Barred annihilation operators F̄_{o,l}
    cre_opts_bar::Matrix{BlockDenseOperator}    # Barred creation operators F̄†_{o,l}
    bare_propagator::BlockDenseDLRPropagator{T} # Bare propagator in DLR representation
    convolution_workspace::ConvolutionWorkspace{T}  # Pre-allocated convolution workspace (max-sized)
end

#=============================================================================
# Format-Specific Storage Types for CalculationData
=============================================================================#

"""
    CalculationDataStorage{T}

Abstract type for format-specific calculation data storage.
Concrete subtypes hold propagator and self-energy in either Dense or BlockDense format.
"""
abstract type CalculationDataStorage{T} end

"""
    DenseCalculationDataStorage{T}

Format-specific storage for DenseFormat calculation data.
"""
struct DenseCalculationDataStorage{T} <: CalculationDataStorage{T}
    propagator::DenseDLRPropagator{T}  # Propagator in DLR representation
    selfenergy::DenseDLRPropagator{T}  # Self-energy in DLR representation
end

"""
    BlockDenseCalculationDataStorage{T}

Format-specific storage for BlockDenseFormat calculation data.
"""
struct BlockDenseCalculationDataStorage{T} <: CalculationDataStorage{T}
    propagator::BlockDenseDLRPropagator{T}  # Propagator in DLR representation
    selfenergy::BlockDenseDLRPropagator{T}  # Self-energy in DLR representation
end

#=============================================================================
# ModelData
=============================================================================#

"""
    ModelData{T, Format, Storage}

Holds the derived data for the model, such as the Hamiltonian, operators, and hybridization function.

The `format_data` field contains format-specific data (operators and bare propagator) in either
Dense or BlockDense representation, eliminating the need for Union{..., Nothing} fields.

# Type Parameters
- `T<:Number`: Numeric type for propagator values (typically ComplexF64)
- `Format<:HilbertSpaceFormat`: Either DenseFormat or BlockDenseFormat
- `Storage<:ModelDataStorage{T}`: Format-specific storage type
"""
struct ModelData{T<:Number, Format<:HilbertSpaceFormat, Storage<:ModelDataStorage{T}}
    # Common fields (same for both formats)
    beta::Float64
    n_flavors::Int
    n_fock::Int
    dlr_grid::DLRGrid
    E_shift::Float64
    n_flavors_physical::Int
    bare_propagator_err::Float64  # Diagnostic: max DLR reconstruction error
    hyb::SOEData
    K0_inverse::Vector{Float64}
    Z_counterterms::Float64
    kernel_cache::KernelCache
    convolution_workspace::ConvolutionWorkspace

    # Format-specific storage (eigenvalues, eigenvectors, operators, bare propagator)
    format_data::Storage
end

#=============================================================================
# CalculationData
=============================================================================#

"""
    CalculationData{T, Format, Storage}

Holds the data for the calculation, such as the Green's function, propagator, etc.

The `format_data` field contains format-specific propagator and self-energy in either
Dense or BlockDense representation.

Note: Green's function is always stored in dense format (it's a physical observable
in small orbital space).

# Type Parameters
- `T<:Number`: Numeric type for propagator values
- `Format<:HilbertSpaceFormat`: Either DenseFormat or BlockDenseFormat
- `Storage<:CalculationDataStorage{T}`: Format-specific storage type
"""
mutable struct CalculationData{T<:Number, Format<:HilbertSpaceFormat, Storage<:CalculationDataStorage{T}}
    # Format-specific storage
    format_data::Storage

    # Always dense (physical observables in small orbital space)
    green::DenseDLRPropagator{T}
    partition_function_by_order::Vector{Float64}
    green_numerator_by_order::Vector{DensePropagator{T}}
    green_by_order::Vector{DensePropagator{T}}
    green_linear_grid::DensePropagator{T}
    tau_linear_grid::Vector{Float64}
end

#=============================================================================
# SolverData
=============================================================================#

"""
    SolverData{T, Format}

Holds all the data required for the solver calculation.
Combines model data and calculation data, both parametrized by the same Format.
"""
struct SolverData{T<:Number, Format<:HilbertSpaceFormat}
    model::ModelData{T, Format}
    calc::CalculationData{T, Format}
end

# Allow SolverData to infer Storage types from its components
function SolverData(model::ModelData{T, Format, MS}, calc::CalculationData{T, Format, CS}) where {T, Format, MS, CS}
    SolverData{T, Format}(model, calc)
end

#=============================================================================
# Type Aliases for Convenience
=============================================================================#

const DenseModelData{T} = ModelData{T, DenseFormat, DenseModelDataStorage{T}}
const BlockDenseModelData{T} = ModelData{T, BlockDenseFormat, BlockDenseModelDataStorage{T}}

const DenseCalculationData{T} = CalculationData{T, DenseFormat, DenseCalculationDataStorage{T}}
const BlockDenseCalculationData{T} = CalculationData{T, BlockDenseFormat, BlockDenseCalculationDataStorage{T}}

#=============================================================================
# Accessor Functions (Unified Interface)
#
# These provide a format-independent way to access format-specific data.
# Simple accessors like these are typically inlined by Julia's compiler (zero overhead).
# For clarity and guaranteed performance, cache the result in a local variable
# when using it multiple times within the same function.
=============================================================================#

# ModelData accessors
get_eigenvalues(model::ModelData) = model.format_data.eigenvalues
get_eigenvectors(model::ModelData) = model.format_data.eigenvectors
get_ann_opts(model::ModelData) = model.format_data.ann_opts
get_cre_opts(model::ModelData) = model.format_data.cre_opts
get_ann_opts_bar(model::ModelData) = model.format_data.ann_opts_bar
get_cre_opts_bar(model::ModelData) = model.format_data.cre_opts_bar
get_bare_propagator(model::ModelData) = model.format_data.bare_propagator

# BlockDense-specific accessor for global convolution workspace
get_blockdense_convolution_workspace(model::ModelData{T, BlockDenseFormat}) where T = model.format_data.convolution_workspace

# CalculationData accessors
get_propagator(calc::CalculationData) = calc.format_data.propagator
get_selfenergy(calc::CalculationData) = calc.format_data.selfenergy
