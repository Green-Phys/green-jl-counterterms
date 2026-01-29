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
        for s1 in 0:(n_spin_impurity - 1)
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
    ### Chemist (Mulliken) notation:
    # H = \sum_{ijkl} U_{ijkl} c^\dagger_i c^\dagger_k c_l c_j
    # Index order: [cre1, ann2, cre2, ann1]
    # No 0.5 prefactor - values are halved to account for antisymmetric pairs

    # Intra-orbital interaction U: U * n_up * n_down
    # Physical term: U * c†_up c†_down c_down c_up
    # Chemist: U[i,j,k,l] c†_i c†_k c_l c_j with i=up, k=down, l=down, j=up -> U[up,up,down,down]
    for o in 1:n_orbital_impurity
        upindex = (o - 1) * n_spin_impurity + 1
        downindex = (o - 1) * n_spin_impurity + 2
        interaction_tensor[upindex, upindex, downindex, downindex] = U / 2
        interaction_tensor[downindex, downindex, upindex, upindex] = U / 2
    end

    # Inter-orbital antiparallel spin interaction U': U' * n_{o1,up} * n_{o2,down} + U' * n_{o1,down} * n_{o2,up}
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                # n_{o1,up} * n_{o2,down}: c†_{up1} c†_{down2} c_{down2} c_{up1}
                interaction_tensor[upindex1, upindex1, downindex2, downindex2] = Uprime / 2
                interaction_tensor[downindex2, downindex2, upindex1, upindex1] = Uprime / 2
                # n_{o1,down} * n_{o2,up}: c†_{down1} c†_{up2} c_{up2} c_{down1}
                interaction_tensor[downindex1, downindex1, upindex2, upindex2] = Uprime / 2
                interaction_tensor[upindex2, upindex2, downindex1, downindex1] = Uprime / 2
            end
        end
    end

    # Inter-orbital parallel spin interaction (U'-J): (U'-J) * n_{o1,σ} * n_{o2,σ}
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                # n_{o1,up} * n_{o2,up}: c†_{up1} c†_{up2} c_{up2} c_{up1}
                interaction_tensor[upindex1, upindex1, upindex2, upindex2] = (Uprime - J) / 2
                interaction_tensor[upindex2, upindex2, upindex1, upindex1] = (Uprime - J) / 2
                # n_{o1,down} * n_{o2,down}: c†_{down1} c†_{down2} c_{down2} c_{down1}
                interaction_tensor[downindex1, downindex1, downindex2, downindex2] = (Uprime - J) / 2
                interaction_tensor[downindex2, downindex2, downindex1, downindex1] = (Uprime - J) / 2
            end
        end
    end

    # Pair hopping J: J * c†_{o1,up} c†_{o1,down} c_{o2,down} c_{o2,up} + h.c.
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                # c†_{up1} c†_{down1} c_{down2} c_{up2}: i=up1, k=down1, l=down2, j=up2 -> U[up1,up2,down1,down2]
                interaction_tensor[upindex1, upindex2, downindex1, downindex2] = J / 2
                interaction_tensor[downindex1, downindex2, upindex1, upindex2] = J / 2
                # h.c.: c†_{up2} c†_{down2} c_{down1} c_{up1}
                interaction_tensor[upindex2, upindex1, downindex2, downindex1] = J / 2
                interaction_tensor[downindex2, downindex1, upindex2, upindex1] = J / 2
            end
        end
    end

    # Spin exchange J: J * c†_{o1,up} c†_{o2,down} c_{o1,down} c_{o2,up} + h.c.
    for o1 in 1:n_orbital_impurity
        for o2 in 1:n_orbital_impurity
            if o1 < o2
                upindex1 = (o1 - 1) * n_spin_impurity + 1
                upindex2 = (o2 - 1) * n_spin_impurity + 1
                downindex1 = (o1 - 1) * n_spin_impurity + 2
                downindex2 = (o2 - 1) * n_spin_impurity + 2
                # c†_{up1} c†_{down2} c_{down1} c_{up2}: i=up1, k=down2, l=down1, j=up2 -> U[up1,up2,down2,down1]
                interaction_tensor[upindex1, upindex2, downindex2, downindex1] = J / 2
                interaction_tensor[downindex2, downindex1, upindex1, upindex2] = J / 2
                # h.c.: c†_{up2} c†_{down1} c_{down2} c_{up1}
                interaction_tensor[upindex2, upindex1, downindex1, downindex2] = J / 2
                interaction_tensor[downindex1, downindex2, upindex2, upindex1] = J / 2
            end
        end
    end

    return interaction_tensor
end

function write_input(file_path::String, impurity_hopping_matrix::Array{Float64, 2}, impurity_interaction_tensor::Array{Float64, 4}, n_orbital_impurity::Int32, n_spin_impurity::Int32)
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
