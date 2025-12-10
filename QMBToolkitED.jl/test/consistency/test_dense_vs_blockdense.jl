"""
Consistency Tests: Dense vs BlockDense Cross-Validation

Validates that Dense and BlockDense methods produce identical numerical results
when given the same physical problem.

These tests compare:
- Operator matrix elements via conversion
- Hamiltonian eigenvalues
- Physical observables (Green's functions tested in global/)
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra

@testset "Consistency: Dense vs BlockDense" begin

    @testset "Operators: Physical Observable Agreement" begin
        nflavors = 4

        # Build operators with both methods
        c_dense = get_annihilators(nflavors)
        basis = make_basis([QN_N(1) for _ in 1:nflavors])
        c_block = get_annihilators_blockdense(basis)

        # Number operator eigenvalues should match
        for i in 1:nflavors
            n_dense = c_dense[i]' * c_dense[i]
            n_block = to_dense(c_block[i]' * c_block[i])

            evals_dense = sort(real(eigvals(n_dense)))
            evals_block = sort(real(eigvals(n_block)))
            @test evals_dense ≈ evals_block atol=1e-12
        end

        # Total particle number should match
        N_dense = sum(c_dense[i]' * c_dense[i] for i in 1:nflavors)
        N_block = to_dense(sum(c_block[i]' * c_block[i] for i in 1:nflavors))

        evals_dense = sort(real(eigvals(N_dense)))
        evals_block = sort(real(eigvals(N_block)))
        @test evals_dense ≈ evals_block atol=1e-12
    end

    @testset "Hamiltonian: Non-interacting System" begin
        hopping_str = """
        0 0 -0.5 0.0
        1 1 -0.3 0.0
        0 1 -1.0 0.0
        1 0 -1.0 0.0
        """
        coulomb_str = "0"

        # Dense method
        H_dense, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
        evals_dense, _ = solve_hamiltonian(H_dense)

        # BlockDense method
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        evals_block, _ = solve_hamiltonian(H_block)

        # Eigenvalues must match exactly
        @test sort(evals_dense) ≈ sort(evals_block) atol=1e-12
    end

    @testset "Hamiltonian: Hubbard Atom" begin
        U = 2.0
        hopping_str = """
        0 0 $(-U/2) 0.0
        1 1 $(-U/2) 0.0
        """
        coulomb_str = """
        2
        0 0 1 1 0 $(U) 0.0
        1 1 0 0 1 $(U) 0.0
        """

        # Dense method
        H_dense, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
        evals_dense, _ = solve_hamiltonian(H_dense)

        # BlockDense method
        site_qns = [QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        evals_block, _ = solve_hamiltonian(H_block)

        # Eigenvalues must match
        @test sort(evals_dense) ≈ sort(evals_block) atol=1e-12
    end

    @testset "Hamiltonian: Hubbard Dimer" begin
        U = 4.0
        t = 1.0
        hopping_str = """
        0 0 $(-U/2) 0.0
        1 1 $(-U/2) 0.0
        2 2 $(-U/2) 0.0
        3 3 $(-U/2) 0.0
        0 2 $(-t) 0.0
        2 0 $(-t) 0.0
        1 3 $(-t) 0.0
        3 1 $(-t) 0.0
        """
        coulomb_str = """
        2
        0 0 1 1 0 $(U) 0.0
        1 1 0 0 1 $(U) 0.0
        2 2 3 3 0 $(U) 0.0
        3 3 2 2 1 $(U) 0.0
        """

        # Dense method
        H_dense, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
        evals_dense, _ = solve_hamiltonian(H_dense)

        # BlockDense method
        site_qns = [QN_N(1), QN_N(1), QN_N(1), QN_N(1)]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        evals_block, _ = solve_hamiltonian(H_block)

        # Eigenvalues must match
        @test sort(evals_dense) ≈ sort(evals_block) atol=1e-12
    end

    @testset "Hamiltonian: Anderson Impurity Model" begin
        # 3-site Anderson impurity model
        ϵ_d = 0.0
        U = 4.0
        V = 1.0
        ϵ_bath = [1.0, -1.0]

        hopping_str = """
        0 0 $(ϵ_d - U/2) 0.0
        1 1 $(ϵ_d - U/2) 0.0
        2 2 $(ϵ_bath[1]) 0.0
        3 3 $(ϵ_bath[1]) 0.0
        4 4 $(ϵ_bath[2]) 0.0
        5 5 $(ϵ_bath[2]) 0.0
        0 2 $(-V) 0.0
        2 0 $(-V) 0.0
        1 3 $(-V) 0.0
        3 1 $(-V) 0.0
        0 4 $(-V) 0.0
        4 0 $(-V) 0.0
        1 5 $(-V) 0.0
        5 1 $(-V) 0.0
        """

        coulomb_str = """
        3
        0 0 1 1 0 $(U) 0.0
        1 1 0 0 1 $(U) 0.0
        2 2 3 3 0 0.0 0.0
        3 3 2 2 1 0.0 0.0
        4 4 5 5 0 0.0 0.0
        5 5 4 4 1 0.0 0.0
        """

        # Dense method
        H_dense, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str))
        evals_dense, _ = solve_hamiltonian(H_dense)

        # BlockDense method
        site_qns = [QN_N(1) for _ in 1:6]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        evals_block, _ = solve_hamiltonian(H_block)

        # Eigenvalues must match
        @test sort(evals_dense) ≈ sort(evals_block) atol=1e-12
    end

end
