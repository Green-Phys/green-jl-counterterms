using DelimitedFiles
using LinearAlgebra

function calculate_Z_counterterms(model_params::ModelParams, n_flavors::Int, n_flavors_physical::Int)
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
    isdiag(unphysical_hopping) || error("Unphysical part of hopping matrix must be diagonal, but found off-diagonal elements")

    # The diagonal elements are the energies of the unphysical sites
    unphysical_energies = real(diag(unphysical_hopping))

    # Partition function Z = product_i (1 + exp(-beta * E_i)) for fermions
    if isempty(unphysical_energies)
        return 1.0
    end
    Z_counterterms = prod(1 .+ exp.(-model_params.beta .* unphysical_energies))
    return Z_counterterms
end