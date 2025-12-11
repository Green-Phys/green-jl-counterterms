
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

struct ModelData{T<:Number}
    beta::Float64
    n_flavors::Int
    n_fock::Int # 2^n_flavors
    ann_opts::Vector{Matrix{T}}
    cre_opts::Vector{Matrix{T}}
    evals::Vector{Float64}
    evecs::Matrix{T}
    dlr_grid::DLRGrid
    E_shift::Float64
    n_flavors_physical::Int
    bare_propagator_err::Float64
    hyb::SOEData
    bare_propagator::DLRData{T}
    K0_inverse::Vector{Float64}
    ann_opts_SOE::Matrix{Matrix{T}} # shape: (n_flavors, p)
    cre_opts_SOE::Matrix{Matrix{T}} # shape: (n_flavors, p)
    Z_counterterms::Float64
    # Performance optimization caches
    kernel_cache::KernelCache  # Pre-computed kernel values for DLR convolution
    convolution_workspace::ConvolutionWorkspace  # Reusable buffers for convolution
end


"""
    CalculationData

Holds the data for the calculation, such as the Green's function, propagator, etc.
"""
mutable struct CalculationData{T<:Number}
    propagator::DLRData{T}
    selfenergy::DLRData{T}
    green::DLRData{T}
    partition_function_by_order::Vector{Float64}  # Z after including each order
    green_numerator_by_order::Vector{Array{T,3}}  # Green's function numerator after each order
    green_by_order::Vector{Array{T,3}}  # Green's function (=numerator/Z) at each order on DLR grid
    green_linear_grid::Array{T,3}  # Green's function on uniform linear tau grid
    tau_linear_grid::Vector{Float64}  # Uniform linear tau grid points
end

"""
    SolverData

Holds all the data required for the solver calculation.
"""
struct SolverData{T<:Number}
    model::ModelData{T}
    calc::CalculationData{T}
end
