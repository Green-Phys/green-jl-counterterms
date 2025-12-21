using QMBToolkitDLR

"""
    ModelData

Holds the derived data for the model, such as the Hamiltonian, operators, and hybridization function.
"""

mutable struct DLRData{T<:Number}
    vals::Array{T,3} # values on the DLR tau grid, first dimension is DLR rank r
    coeffs::Array{T,3}

    # --- Constructors ---

    function DLRData(vals::Array{T,3}, coeffs::Array{T,3}) where T<:Number
        return new{T}(vals, coeffs)
    end

    function DLRData{T}(r::Int, flavor1::Int, flavor2::Int) where T<:Number
        vals = zeros(T, r, flavor1, flavor2)
        coeffs = zeros(T, r, flavor1, flavor2)
        return new{T}(vals, coeffs)
    end

    function DLRData(::Nothing, coeffs::Array{T,3}) where T<:Number
        vals = zeros(T, size(coeffs))
        return new{T}(vals, coeffs)
    end

    function DLRData(vals::Array{T,3}, ::Nothing) where T<:Number
        coeffs = zeros(T, size(vals))
        return new{T}(vals, coeffs)
    end
end

# Convenience constructor that infers type parameter
DLRData(r::Int, flavor1::Int, flavor2::Int) = DLRData{Float64}(r, flavor1, flavor2)

mutable struct SOEData # the value is recovered by sum_{i} coeffs[i] * K(tau, exponents[i]), different matrix element share the same exponents
    coeffs::Array{Float64,3} # SOE coefficients, shape: (p, o1, o2)
    exponents::Vector{Float64} # SOE exponents, shape: (p,)
    KPlus_on_DLRtau::Array{Float64,2} # K(tau_k, exponent_i) on the DLR tau grid
    KMinus_on_DLRtau::Array{Float64,2} # K(tau_k, -exponent_i) on the DLR tau grid
end

struct ModelData{T<:Number, Format<:HilbertSpaceFormat}
    beta::Float64
    n_flavors::Int
    n_fock::Int # 2^n_flavors
    # Dense format fields (populated for DenseFormat, nothing for BlockDenseFormat)
    ann_opts::Union{Vector{Matrix{T}}, Nothing}  # Dense annihilation operators
    cre_opts::Union{Vector{Matrix{T}}, Nothing}  # Dense creation operators
    evals::Vector{Float64}
    evecs::Matrix{T}
    dlr_grid::DLRGrid
    E_shift::Float64
    n_flavors_physical::Int
    bare_propagator_err::Float64
    hyb::SOEData
    bare_propagator::DLRData{T}  # Dense bare propagator DLR representation
    K0_inverse::Vector{Float64}
    ann_opts_SOE::Union{Matrix{Matrix{T}}, Nothing}  # Dense SOE operators (nothing for BlockDenseFormat)
    cre_opts_SOE::Union{Matrix{Matrix{T}}, Nothing}  # Dense SOE operators (nothing for BlockDenseFormat)
    Z_counterterms::Float64
    # Performance optimization caches
    kernel_cache::KernelCache  # Pre-computed kernel values for DLR convolution
    convolution_workspace::ConvolutionWorkspace  # Reusable buffers for convolution (dense)
    # BlockDense format fields (only populated for BlockDenseFormat)
    bare_propagator_blockdense::Any  # BlockDensePropagator{T} for BlockDenseFormat, nothing for DenseFormat
    cre_opts_blockdense::Any  # Vector{BlockDenseOperator} for BlockDenseFormat
    ann_opts_blockdense::Any  # Vector{BlockDenseOperator} for BlockDenseFormat
    cre_opts_SOE_blockdense::Any  # Matrix{BlockDenseOperator} for BlockDenseFormat
    ann_opts_SOE_blockdense::Any  # Matrix{BlockDenseOperator} for BlockDenseFormat
end


"""
    CalculationData

Holds the data for the calculation, such as the Green's function, propagator, etc.

Note: Green's function is always stored in dense format (it's a physical observable
in small orbital space), but propagator and self-energy can be format-dependent.

# Format-dependent storage
- For DenseFormat: uses propagator_dense and selfenergy_dense (DLRData{T})
- For BlockDenseFormat: uses propagator_blockdense and selfenergy_blockdense (BlockDensePropagator{T})
- Green's function is always dense (physical observable in small orbital space)
"""
mutable struct CalculationData{T<:Number, Format<:HilbertSpaceFormat}
    # Dense format fields
    propagator_dense::Union{DLRData{T}, Nothing}  # Only for DenseFormat
    selfenergy_dense::Union{DLRData{T}, Nothing}  # Only for DenseFormat
    # BlockDense format fields
    propagator_blockdense::Any  # BlockDensePropagator{T} for BlockDenseFormat, nothing for DenseFormat
    selfenergy_blockdense::Any  # BlockDensePropagator{T} for BlockDenseFormat, nothing for DenseFormat
    # Always dense (physical observables)
    green::DLRData{T}  # Always dense (physical observable, small orbital space)
    partition_function_by_order::Vector{Float64}  # Z after including each order
    green_numerator_by_order::Vector{Array{T,3}}  # Green's function numerator after each order (always dense)
    green_by_order::Vector{Array{T,3}}  # Green's function (=numerator/Z) at each order on DLR grid (always dense)
    green_linear_grid::Array{T,3}  # Green's function on uniform linear tau grid (always dense)
    tau_linear_grid::Vector{Float64}  # Uniform linear tau grid points
end

"""
    SolverData

Holds all the data required for the solver calculation.

Combines model data and calculation data, both parametrized by the same Format.
"""
struct SolverData{T<:Number, Format<:HilbertSpaceFormat}
    model::ModelData{T, Format}
    calc::CalculationData{T, Format}
end
