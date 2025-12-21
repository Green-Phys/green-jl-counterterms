module GlobalTestUtils

using Test
using QMBToolkitED
using QMBToolkitED.Dense
using QMBToolkitED.BlockDense
using LinearAlgebra

export run_hamiltonian_test, run_greens_test, verify_anticommutation

"""
    run_hamiltonian_test(test_name; hopping_str, coulomb_str, site_qns_n, site_qns_sz, verify_fn)

Runs a Hamiltonian test across Dense, BlockDense (QN_N), and BlockDense (QN_NSz) backends.
`verify_fn` should take `(H, annh, method_label)` as arguments.
"""
function run_hamiltonian_test(test_name; hopping_str, coulomb_str, site_qns_n, site_qns_sz, verify_fn)
    @testset "$test_name (Dense)" begin
        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:dense)
        verify_fn(H, annh, "Dense")
    end

    @testset "$test_name (BlockDense QN_N)" begin
        basis = make_basis(site_qns_n)
        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        verify_fn(H, annh, "BlockDense N")
    end

    @testset "$test_name (BlockDense QN_NSz)" begin
        basis = make_basis(site_qns_sz)
        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        verify_fn(H, annh, "BlockDense NSz")
    end
end

"""
    run_greens_test(test_name; hopping_str, coulomb_str, site_qns_n, site_qns_sz, beta, verify_fn)

Runs a Green's function test across Dense, BlockDense (QN_N), and BlockDense (QN_NSz) backends.
`verify_fn` should take `(result, annh)` as arguments, where `result` is the solver output.
"""
function run_greens_test(test_name; hopping_str, coulomb_str, site_qns_n, site_qns_sz, verify_fn)
    @testset "$test_name (Dense)" begin
        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:dense)
        result = solve_hamiltonian(H)
        verify_fn(result, annh)
    end

    @testset "$test_name (BlockDense QN_N)" begin
        basis = make_basis(site_qns_n)
        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        result = solve_hamiltonian(H)
        verify_fn(result, annh)
    end

    @testset "$test_name (BlockDense QN_NSz)" begin
        basis = make_basis(site_qns_sz)
        H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        result = solve_hamiltonian(H)
        verify_fn(result, annh)
    end
end

"""
    verify_anticommutation(c, c_dag, dim; atol=1e-9)

Verifies fermionic anticommutation relations:
{c_i, c_j} = 0
{c_i, c_j^†} = δ_ij * I
"""
function verify_anticommutation(c, c_dag, dim; atol=1e-9)
    nflavors = length(c)
    
    # Helper to convert to raw matrix (use the unified to_matrix function)
    _to_mat(op::DenseOperator) = to_matrix(op)
    _to_mat(op::BlockDenseOperator) = to_matrix(op)
    _to_mat(op::Matrix) = op

    # Test {c_i, c_j} = 0
    for i in 1:nflavors
        for j in (i+1):nflavors
            res = _to_mat(c[i] * c[j] + c[j] * c[i])
            @test res ≈ zeros(ComplexF64, dim, dim) atol=atol
        end
    end

    # Test {c_i, c_j†} = δ_ij * I
    for i in 1:nflavors
        for j in 1:nflavors
            res = _to_mat(c[i] * c_dag[j] + c_dag[j] * c[i])
            if i == j
                @test res ≈ Matrix{ComplexF64}(I, dim, dim) atol=atol
            else
                @test res ≈ zeros(ComplexF64, dim, dim) atol=atol
            end
        end
    end
end

end # module
