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


    @testset "Performance Benchmark (6 sites)" begin # Changed from 8 to 6 sites for faster testing
        # 6 sites = 12 orbitals (spinful) -> 2^12 = 4096 states
        # This is a manageable size to show performance differences.
        
        n_sites = 6
        n_flavors = 2 * n_sites
        t = 1.0
        U = 4.0
        
        # Construct inputs
        hopping_io = IOBuffer()
        for i in 0:(n_sites-2)
            # Hopping between i and i+1
            # Up spin (even indices)s)
            println(hopping_io, "$(2*i) $(2*(i+1)) $(-t) 0.0")
            println(hopping_io, "$(2*(i+1)) $(2*i) $(-t) 0.0")
            # Down spin (odd indices)
            println(hopping_io, "$(2*i+1) $(2*(i+1)+1) $(-t) 0.0")
            println(hopping_io, "$(2*(i+1)+1) $(2*i+1) $(-t) 0.0")
        end
        # On-site energies
        for i in 0:(2*n_sites-1)
            println(hopping_io, "$i $i $(-U/2) 0.0")
        end
        
        coulomb_io = IOBuffer()
        println(coulomb_io, "$n_sites") # One interaction per site
        for i in 0:(n_sites-1)
            # Up-Down interaction on site i
            # Up index (0-indexed): 2*i, Down index (0-indexed): 2*i+1
            # Term: c_up' c_down' c_down c_up
            println(coulomb_io, "$i $(2*i) $(2*i+1) $(2*i+1) $(2*i) $U 0.0")
        end
        
        hopping_str = String(take!(hopping_io))
        coulomb_str = String(take!(coulomb_io))
        
        println("\n--- Performance Benchmark: 1D Hubbard Chain ($n_sites sites, $n_flavors orbitals) ---")
        println("Hilbert Space Dimension: $(2^n_flavors)")
        
        # 1. Theoretical Estimate
        total_dim = 2^n_flavors
        # Calculate theoretical speedup based on sum of cubes of sector dimensions
        
        # We need site_qns for the basis. Let's use QN_NSz for Hubbard, so 2 states per site
        site_qns = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:n_flavors]
        basis = make_basis(site_qns)
        
        sum_dim_cubed_theory = sum(Float64(s.dim)^3 for s in basis.sectors)
        theory_speedup = (Float64(total_dim)^3) / sum_dim_cubed_theory
        println("Theoretical Speedup (sum d_i^3 vs D^3): $(round(theory_speedup, digits=2))x")
        
        # 2. BlockDense Timing
        # Build operator
H_block, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)
        
        println("Running BlockDense diagonalization...")
        # Compile run
        solve_hamiltonian(H_block)
        
        # Timed run
        stats_block = @timed solve_hamiltonian(H_block)
        time_block = stats_block.time
        mem_block = stats_block.bytes / 1024^2 # MB
        
        println("BlockDense: $(round(time_block, digits=4)) s, $(round(mem_block, digits=2)) MB")
        
        # 3. Dense Timing
        # We will use 4 sites = 8 orbitals -> 256 states. This is suitable for direct comparison.
        n_sites_comp = 4
        n_flavors_comp = 2 * n_sites_comp
        println("\n--- Comparative Benchmark: 1D Hubbard Chain ($n_sites_comp sites, $n_flavors_comp orbitals) ---")
        println("Hilbert Space Dimension: $(2^n_flavors_comp) (256)")
        
        # Re-generate inputs for smaller system
        hopping_io_comp = IOBuffer()
        for i in 0:(n_sites_comp-2)
            println(hopping_io_comp, "$(2*i) $(2*(i+1)) $(-t) 0.0")
            println(hopping_io_comp, "$(2*(i+1)) $(2*i) $(-t) 0.0")
            println(hopping_io_comp, "$(2*i+1) $(2*(i+1)+1) $(-t) 0.0")
            println(hopping_io_comp, "$(2*(i+1)+1) $(2*i+1) $(-t) 0.0")
        end
        for i in 0:(2*n_sites_comp-1)
            println(hopping_io_comp, "$i $i $(-U/2) 0.0")
        end
        coulomb_io_comp = IOBuffer()
        println(coulomb_io_comp, "$n_sites_comp")
        for i in 0:(n_sites_comp-1)
            # Up-Down interaction on site i
            # Up index (0-indexed): 2*i, Down index (0-indexed): 2*i+1
            # Term: c_up' c_down' c_down c_up
            println(coulomb_io_comp, "$i $(2*i) $(2*i+1) $(2*i+1) $(2*i) $U 0.0")
        end
        
        hop_str_c = String(take!(hopping_io_comp))
        coul_str_c = String(take!(coulomb_io_comp))
        
        # Dense Run
        H_dense_c, _ = build_hamiltonian(IOBuffer(hop_str_c), IOBuffer(coul_str_c))
        println("Running Dense diagonalization...")
        solve_hamiltonian(H_dense_c) # Compile
        stats_dense = @timed solve_hamiltonian(H_dense_c)
        
        # BlockDense Run
        site_qns_c = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:n_flavors_comp]
        basis_c = make_basis(site_qns_c)
        H_block_c, _ = build_hamiltonian(IOBuffer(hop_str_c), IOBuffer(coul_str_c); method=:blockdense, basis=basis_c)
        println("Running BlockDense diagonalization...")
        solve_hamiltonian(H_block_c) # Compile
        stats_block_c = @timed solve_hamiltonian(H_block_c)
        
        # Results
        t_dense = stats_dense.time
        t_block = stats_block_c.time
        m_dense = stats_dense.bytes / 1024^2
        m_block = stats_block_c.bytes / 1024^2
        
        println("Dense:      $(round(t_dense, digits=4)) s, $(round(m_dense, digits=2)) MB")
        println("BlockDense: $(round(t_block, digits=4)) s, $(round(m_block, digits=2)) MB")
        
        speedup = t_dense / t_block
        mem_saving = m_dense / m_block
        
        println("Speedup: $(round(speedup, digits=2))x")
        println("Memory Saving: $(round(mem_saving, digits=2))x")
        
        # Theoretical speedup calculation for comparative benchmark
        total_dim_c = 2^n_flavors_comp
        sum_dim_cubed_c = sum(Float64(s.dim)^3 for s in basis_c.sectors)
        theory_speedup_comp = (Float64(total_dim_c)^3) / sum_dim_cubed_c
        println("Theoretical Speedup (N=$(n_flavors_comp)): $(round(theory_speedup_comp, digits=2))x")
        
        @test speedup > 1.0 # Should be faster
    end
end
