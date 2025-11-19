using ArgParse
using Printf
using LinearAlgebra


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--n_spin_impurity"
        help = "number of spin flavors for the impurity"
        arg_type = Int32
        required = true
        "--n_orbital_impurity"
        help = "number of orbitals for the impurity"
        arg_type = Int32
        required = true
        "--t"
        help = "hopping amplitude"
        arg_type = Float64
        required = true
        "--U"
        help = "Coulomb interaction"
        arg_type = Float64
        required = true
        "--J"
        help = "Hund's coupling"
        arg_type = Float64
        required = true
        "--Uprime"
        help = "inter-orbital Coulomb interaction"
        arg_type = Float64
        required = true
        "--mu"
        help = "chemical potential"
        arg_type = Float64
        required = true
        "--output_dir"
        help = "Output directory path"
        arg_type = String
        required = true
    end
    return parse_args(s)
end

function print_input(args)
    println("--- Input parameters ---")
    println("------------------")
    println(@sprintf("Number of impurity orbitals: %d", args["n_orbital_impurity"]))
    println(@sprintf("Number of impurity spin flavors: %d", args["n_spin_impurity"]))
    println(@sprintf("Hopping amplitude t: %.6f", args["t"]))
    println(@sprintf("Coulomb interaction U: %.6f", args["U"]))
    println(@sprintf("Hund's coupling J: %.6f", args["J"]))
    println(@sprintf("Inter-orbital Coulomb interaction U': %.6f", args["Uprime"]))
    println(@sprintf("Chemical potential mu: %.6f", args["mu"]))
    println(@sprintf("Output directory: %s", args["output_dir"]))
end


function generate_impurity_hopping_matrix(t::Float64, mu::Float64, n_orbital_impurity::Int32, n_spin_impurity::Int32)
    @assert n_spin_impurity == 2 || n_spin_impurity == 1 "Currently only spin-1/2 or spinless fermions are supported."
    n_flavors_impurity = n_orbital_impurity * n_spin_impurity
    impurity_hopping_matrix = zeros(Float64, n_flavors_impurity, n_flavors_impurity)
    for o1 in 1:n_orbital_impurity
        for s1 in 0:(n_spin_impurity-1)
            flavor_i = (o1 - 1) * n_spin_impurity + s1 + 1
            # on-site energy with chemical potential
            impurity_hopping_matrix[flavor_i, flavor_i] = -mu
            # hopping between orbitals (assuming only nearest neighbor hopping with amplitude t)
            for o2 in 1:n_orbital_impurity
                if o1 != o2
                    # hopping only between same spin flavors
                    flavor_j = (o2 - 1) * n_spin_impurity + s1 + 1
                    impurity_hopping_matrix[flavor_i, flavor_j] = -t
                end
            end
        end
    end

    return impurity_hopping_matrix
end

function generate_impurity_interaction_tensor(U::Float64, J::Float64, Uprime::Float64, n_orbital_impurity::Int32, n_spin_impurity::Int32)
    n_flavors = n_orbital_impurity * n_spin_impurity
    interaction_tensor = zeros(Float64, n_flavors, n_flavors, n_flavors, n_flavors)
    if n_spin_impurity == 1
        # trivial case for spinless fermions, return zero tensor
        return interaction_tensor
    elseif n_spin_impurity != 2
        error("Currently only spin-1/2 or spinless fermions are supported.")
    end
    ### We assume the order of the flavor is (orbital1_up, orbital1_down, orbital2_up, orbital2_down, ...)
    ### The notation for local interaction in ALPSCore is
    #H = 0.5 \sum_{ijkl} U_{ijkl} c^\dagger_i c^\dagger_j c_k c_l; here interaction_tensor is U_{ijkl}

    # Intra-orbital interaction U
    for o in 1:n_orbital_impurity
        upindex = (o - 1) * n_spin_impurity + 1
        downindex = (o - 1) * n_spin_impurity + 2
        interaction_tensor[upindex, downindex, downindex, upindex] = U
        interaction_tensor[downindex, upindex, upindex, downindex] = U
    end

    # Inter-orbital antiparallel spin interaction U'
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                interaction_tensor[upindex1, downindex2, downindex2, upindex1] = Uprime
                interaction_tensor[downindex2, upindex1, upindex1, downindex2] = Uprime
                interaction_tensor[downindex1, upindex2, upindex2, downindex1] = Uprime
                interaction_tensor[upindex2, downindex1, downindex1, upindex2] = Uprime
            end
        end
    end

    # Inter-orbital parallel spin interaction U'-J
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                interaction_tensor[upindex1, upindex2, upindex2, upindex1] = (Uprime - J)
                interaction_tensor[upindex2, upindex1, upindex1, upindex2] = (Uprime - J)
                interaction_tensor[downindex1, downindex2, downindex2, downindex1] = (Uprime - J)
                interaction_tensor[downindex2, downindex1, downindex1, downindex2] = (Uprime - J)
            end
        end
    end

    # pairing hopping J
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                interaction_tensor[upindex1, downindex1, downindex2, upindex2] = J
                interaction_tensor[downindex1, upindex1, upindex2, downindex2] = J
                # conjugate terms
                interaction_tensor[upindex2, downindex2, downindex1, upindex1] = J
                interaction_tensor[downindex2, upindex2, upindex1, downindex1] = J
            end
        end
    end

    # spin exchange J
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                interaction_tensor[upindex1, downindex2, downindex1, upindex2] = J
                interaction_tensor[downindex2, upindex1, upindex2, downindex1] = J
                # conjugate terms
                interaction_tensor[upindex2, downindex1, downindex2, upindex1] = J
                interaction_tensor[downindex1, upindex2, upindex1, downindex2] = J
            end
        end
    end

    return interaction_tensor
end

function write_input(file_path::String, impurity_hopping_matrix::Array{Float64,2}, impurity_interaction_tensor::Array{Float64,4}, n_orbital_impurity::Int32, n_spin_impurity::Int32)
    n_flavors_impurity = n_orbital_impurity * n_spin_impurity
    # write hopping matrix to file_path/hopping_matrix.txt
    # 1st column: the flavor index i (from 0 to n_flavors_impurity-1),
    # 2nd column: the flavor index j (from 0 to n_flavors_impurity-1),
    # 3rd column: real part of hopping,
    # 4th column: imaginary part of hopping (0.0 here)
    @assert all(size(impurity_hopping_matrix) .== n_orbital_impurity * n_spin_impurity)
    open(joinpath(file_path, "hopping.txt"), "w") do file
        for i in 1:(n_flavors_impurity)
            for j in 1:(n_flavors_impurity)
                line = @sprintf("%d %d %.16f %.1f\n", i - 1, j - 1, real(impurity_hopping_matrix[i, j]), imag(impurity_hopping_matrix[i, j]))
                write(file, line)
            end
        end
    end

    # write interaction tensor to file_path/interaction_tensor.txt
    # 1st row: the number of non-zero elements
    # from the second row:
    #	1st column: the index of non-zero elements in the tensor (from 0 in increasing order),
    #	2nd column: the flavor index i (from 0 to n_flavors_impurity-1),
    #	3rd column: the flavor index j (from 0 to n_flavors_impurity-1),
    #	4th column: the flavor index k (from 0 to n_flavors_impurity-1),
    #	5th column: the flavor index l (from 0 to n_flavors_impurity-1),
    #	6th column: real part of U_{ijkl},
    #7th column: imaginary part of U_{ijkl} (0.0 here).
    # check the number of non-zero elements
    @assert all(size(impurity_interaction_tensor) .== n_flavors_impurity)
    tol = 1e-14
    nnz = count(x -> abs(x) > tol, impurity_interaction_tensor)
    open(joinpath(file_path, "Uijkl.txt"), "w") do file
        line = @sprintf("%d\n", nnz)
        write(file, line)
        idx = 0
        for i in 1:size(impurity_interaction_tensor, 1)
            for j in 1:size(impurity_interaction_tensor, 2)
                for k in 1:size(impurity_interaction_tensor, 3)
                    for l in 1:size(impurity_interaction_tensor, 4)
                        if abs(impurity_interaction_tensor[i, j, k, l]) > tol
                            line = @sprintf("%d %d %d %d %d %.16f %.1f\n", idx, i - 1, j - 1, k - 1, l - 1, real(impurity_interaction_tensor[i, j, k, l]), imag(impurity_interaction_tensor[i, j, k, l]))
                            write(file, line)
                            idx += 1
                        end
                    end
                end
            end
        end
        @assert idx == nnz
    end

end

function main()
    println("=== Generating impurity hopping matrix and interaction tensor ===")
    args = parse_commandline()
    print_input(args)
    impurity_hopping_matrix = generate_impurity_hopping_matrix(args["t"], args["mu"], args["n_orbital_impurity"], args["n_spin_impurity"])
    impurity_interaction_tensor = generate_impurity_interaction_tensor(args["U"], args["J"], args["Uprime"], args["n_orbital_impurity"], args["n_spin_impurity"])
    if !isdir(args["output_dir"])
        mkdir(args["output_dir"])
    end
    write_input(args["output_dir"], impurity_hopping_matrix, impurity_interaction_tensor, args["n_orbital_impurity"], args["n_spin_impurity"])
    println("--- Finished ---")
end

main()
