using SparseArrays
using QMBToolkitDLR

"""
    ModelData

Holds the derived data for the model, such as the Hamiltonian, operators, and hybridization function.
"""
struct ModelData
    poles_energies::Vector{Float64}
    poles_hybs::Matrix{Float64}
    beta::Float64
    n_flavors::Int
    n_fock::Int # 2^n_flavors
    ann_opts::Vector{SparseMatrixCSC{Int8,Int}}
    cre_opts::Vector{SparseMatrixCSC{Int8,Int}}
    evals::Vector{Float64}
    evecs::Matrix{ComplexF64}
    hyb::Array{Float64,3}
    tau_grid_hyb::Array{Float64,1}
    dlr_grid::DLRGrid
    bare_propagator::Array{Float64,3}
    E_shift::Float64
    n_flavors_physical::Int
    n_fock_physical::Int
end


mutable struct DLRData
    vals::Array{Float64,3} # values on the DLR tau grid
    coeffs::Array{Float64,3}

    # --- Constructors ---
    
    function DLRData(vals::Array{Float64,3}, coeffs::Array{Float64,3})
        return new(vals, coeffs)
    end
s
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

"""
    CalculationData

Holds the data for the calculation, such as the Green's function, propagator, etc.
"""
mutable struct CalculationData
    hyb_error::Float64
    bare_propagator_err::Float64
    hyb::DLRData
    propagator::DLRData
    bare_propagator::DLRData
    selfenergy::DLRData
    green::DLRData
end

"""
    SolverData

Holds all the data required for the solver calculation.
"""
struct SolverData
    model::ModelData
    calc::CalculationData
end
