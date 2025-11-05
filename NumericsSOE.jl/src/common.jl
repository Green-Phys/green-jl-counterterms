export AbstractExponentMethod, AbstractCoefficientMethod, CommonOptions

abstract type AbstractExponentMethod end
abstract type AbstractCoefficientMethod end

"""
    CommonOptions

A struct for holding common options across different methods.

# Fields
- `verbose::Bool`: If true, enables verbose logging and warnings.
"""
struct CommonOptions
    verbose::Bool
end

# Keyword constructor with default values
CommonOptions(; verbose::Bool=false) = CommonOptions(verbose)
