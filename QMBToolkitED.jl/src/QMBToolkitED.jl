module QMBToolkitED

using LinearAlgebra, SparseArrays

export get_annihilators, build_hamiltonian, solve_hamiltonian, calculate_greens_function

"""
    get_annihilators(nflavours::Int)

Generate fermionic annihilation operators for a Fock space with `nflavours`.

The operators are represented as sparse matrices in the occupation number basis.
The Fock space is ordered such that the i-th bit of an integer corresponds to the occupation of the i-th state. E.g., for `nflavours = 3`, the state |100> has the first state occupied and the others empty, the corresponding integer (fock_state) is 4.

"""
function get_annihilators(nflavours::Int)
    nfock = 1 << nflavours

    # Annihilation operators
    c = Vector{SparseMatrixCSC{Int8,Int}}(undef, nflavours)

    # Sign for fermionic anti-commutation relations
    sign = ones(Int8, nfock)

    for i in 1:nflavours
        bitmask = 1 << (i - 1)

        # This is exactly half the states, so we can pre-allocate.
        num_non_zero = nfock >> 1
        I = Vector{Int}(undef, num_non_zero)
        J = Vector{Int}(undef, num_non_zero)
        V = Vector{Int8}(undef, num_non_zero)
        count = 0

        for alpha in 1:nfock
            fock_state = alpha - 1 # 0-indexed fock state
            if (fock_state & bitmask) != 0
                count += 1
                # State i is occupied, can be annihilated
                I[count] = (fock_state - bitmask) + 1 
                J[count] = alpha
                V[count] = sign[alpha]

                # Update sign for the next operator c_{i+1}
                sign[alpha] *= -1
            end
        end
        c[i] = sparse(I, J, V, nfock, nfock)
    end

    return c
end

"""
    build_hamiltonian(hopping_matrix_input_file::String, coulomb_tensor_input_file::String)

Construct the Hamiltonian from input files.

Parameters:
- `hopping_matrix_input_file`: Path to the file containing the hopping matrix.
- `coulomb_tensor_input_file`: Path to the file containing the Coulomb tensor.

Returns the Hamiltonian matrix and the annihilation operators.
"""
function build_hamiltonian(hopping_matrix_input_file::String, coulomb_tensor_input_file::String)
    # Read hopping matrix
    hopping_lines = readlines(hopping_matrix_input_file)
    max_flavor = 0
    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            max_flavor = max(max_flavor, i, j)
        end
    end
    nflavours = max_flavor + 1
    nfock = 1 << nflavours

    annh = get_annihilators(nflavours)
    
    H_hop = spzeros(ComplexF64, nfock, nfock)
    if !isempty(hopping_lines)
        for line in hopping_lines
            parts = split(line)
            i = parse(Int, parts[1]) + 1
            j = parse(Int, parts[2]) + 1
            val = parse(Float64, parts[3]) + im * parse(Float64, parts[4])
            H_hop += val * (annh[i]' * annh[j])
        end
    end

    # Read Coulomb tensor
    coulomb_lines = readlines(coulomb_tensor_input_file)
    H_int = spzeros(ComplexF64, nfock, nfock)
    if length(coulomb_lines) > 1
        for line in coulomb_lines[2:end]
            parts = split(line)
            i = parse(Int, parts[2]) + 1
            j = parse(Int, parts[3]) + 1
            k = parse(Int, parts[4]) + 1
            l = parse(Int, parts[5]) + 1
            val = parse(Float64, parts[6]) + im * parse(Float64, parts[7])
            H_int += 0.5 * val * (annh[i]' * annh[j]' * annh[k] * annh[l])
        end
    end

    H = H_hop + H_int
    return H, annh
end


"""
    solve_hamiltonian(H::SparseMatrixCSC)

Diagonalize the Hamiltonian to find eigenvalues and eigenvectors.
"""
function solve_hamiltonian(H::SparseMatrixCSC)
    H_dense = Matrix(H)
    # Ensure H is treated as Hermitian
    sol = eigen(Hermitian(H_dense))
    return sol.values, sol.vectors
end

"""
    calculate_greens_function(evals, evecs, annh_ops, beta, n_tau, i, j)

Compute an element of the imaginary-time Green's function matrix G(τ).
Specifically, it calculates -<c_i^†(τ) c_j(0)>.
"""
function calculate_greens_function(evals::Vector{<:Real}, evecs::Matrix, annh_ops, beta::Real, n_tau::Int, i::Int, j::Int)
    # Get annihilation operators for indices i and j
    c_i = annh_ops[i]
    c_j = annh_ops[j]
    c_i_dag = c_i'

    # Transform operators to the eigenbasis of the Hamiltonian
    c_i_dag_tilde = evecs' * Matrix(c_i_dag) * evecs
    c_j_tilde = evecs' * Matrix(c_j) * evecs

    # Shift energies so that ground state is at 0
    evals .-= minimum(evals)

    # Partition function
    Z = sum(exp.(-beta .* evals))

    tau_grid = range(0, beta, length=n_tau)
    gtau = zeros(ComplexF64, n_tau)

    evolve(tau, e) = Diagonal(exp.(-tau .* e))

    for (k, t) in enumerate(tau_grid)
        D1 = evolve(beta - t, evals)
        D2 = evolve(t, evals)
        # Calculates Tr(exp(-(β-τ)H) c_i† exp(-τH) c_j)
        gtau[k] = tr(D1 * c_i_dag_tilde * D2 * c_j_tilde)
    end

    gtau ./= -Z

    return tau_grid, gtau
end

end # module QMBToolkitED