using QuadGK

function compute_Z_Bethe(beta)
    # Define the integrand: ln(1 + exp(-2βcos(2πx)))
    integrand = x -> log(1 + exp(-2 * beta * cos(2 * pi * x)))
    
    # Compute the definite integral from 0 to 1
    # quadgk returns a tuple (result, error_estimate), 
    # so we take the first element [1]
    integral_result = quadgk(integrand, 0, 1)[1]
    
    # Return the exponential of the integral's result
    return exp(integral_result)
end

function compute_Z_noninteracting(beta, energies)
    Z = 1.0
    for ε in energies
        Z *= (1 + exp(-beta * ε))
    end
    return Z
end