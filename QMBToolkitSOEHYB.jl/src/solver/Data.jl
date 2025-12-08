
using QMBToolkitDLR

"""
    ModelData

Holds the derived data for the model, such as the Hamiltonian, operators, and hybridization function.
"""

mutable struct DLRData
    vals::Array{Float64,3} # values on the DLR tau grid, first dimension is DLR rank r
    coeffs::Array{Float64,3}

    # --- Constructors ---
    
    function DLRData(vals::Array{Float64,3}, coeffs::Array{Float64,3})
        return new(vals, coeffs)
    end

    function DLRData(r::Int, flavor1::Int, flavor2::Int)
        vals = zeros(Float64, r, flavor1, flavor2)
        coeffs = zeros(Float64, r, flavor1, flavor2)
        return new(vals, coeffs)
    end

    function DLRData(::Nothing, coeffs::Array{Float64,3})
        vals = zeros(Float64, size(coeffs))
        return new(vals, coeffs)
    end

    function DLRData(vals::Array{Float64,3}, ::Nothing)
        coeffs = zeros(Float64, size(vals))
        return new(vals, coeffs)
    end
end

mutable struct SOEData # the value is recovered by sum_{i} coeffs[i] * K(tau, exponents[i]), different matrix element share the same exponents
    coeffs::Array{Float64,3} # SOE coefficients, shape: (p, o1, o2)
    exponents::Vector{Float64} # SOE exponents, shape: (p,)
    KPlus_on_DLRtau::Array{Float64,2} # K(tau_k, exponent_i) on the DLR tau grid
    KMinus_on_DLRtau::Array{Float64,2} # K(tau_k, -exponent_i) on the DLR tau grid
end

struct ModelData
    beta::Float64
    n_flavors::Int
    n_fock::Int # 2^n_flavors
    ann_opts::Vector{Matrix{Float64}}
    cre_opts::Vector{Matrix{Float64}}
    evals::Vector{Float64}
    evecs::Matrix{ComplexF64}
    dlr_grid::DLRGrid
    E_shift::Float64
    n_flavors_physical::Int
    bare_propagator_err::Float64
    hyb::SOEData
    bare_propagator::DLRData
    K0_inverse::Vector{Float64}
    ann_opts_SOE::Matrix{Matrix{Float64}} # shape: (n_flavors, p)
    cre_opts_SOE::Matrix{Matrix{Float64}} # shape: (n_flavors, p)
    Z_counterterms::Float64
    # Performance optimization caches
    kernel_cache::KernelCache  # Pre-computed kernel values for DLR convolution
    convolution_workspace::ConvolutionWorkspace  # Reusable buffers for convolution
end


"""
    CalculationData

Holds the data for the calculation, such as the Green's function, propagator, etc.
"""
mutable struct CalculationData
    propagator::DLRData
    selfenergy::DLRData
    green::DLRData
    partition_function_by_order::Vector{Float64}  # Z after including each order
    green_numerator_by_order::Vector{Array{Float64,3}}  # Green's function numerator after each order
    green_by_order::Vector{Array{Float64,3}}  # Green's function (=numerator/Z) at each order on DLR grid
    green_linear_grid::Array{Float64,3}  # Green's function on uniform linear tau grid
    tau_linear_grid::Vector{Float64}  # Uniform linear tau grid points
end

"""
    SolverData

Holds all the data required for the solver calculation.
"""
struct SolverData
    model::ModelData
    calc::CalculationData
end
