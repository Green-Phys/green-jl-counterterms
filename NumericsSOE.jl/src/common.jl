export AbstractExponentMethod, AbstractCoefficientMethod, AbstractSOEOutput, SOEOutput, AbstractExponentsOutput

abstract type AbstractSOEOutput end
abstract type AbstractExponentsOutput end

struct SOEOutput{S, R, E <: AbstractExponentsOutput} <: AbstractSOEOutput
    s::S
    R::R
    exponents_output::E
end

abstract type AbstractExponentMethod end
abstract type AbstractCoefficientMethod end
struct CommonOptions
    verbose::Bool
end

# Keyword constructor with default values
CommonOptions(; verbose::Bool=false) = CommonOptions(verbose)
