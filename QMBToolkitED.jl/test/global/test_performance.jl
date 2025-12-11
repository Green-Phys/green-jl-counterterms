"""
Performance Benchmark Tests

Compares Dense vs BlockDense performance on moderately-sized systems.
Includes benchmarks for:
1. Diagonalization (Eigen decomposition)
2. Operator Multiplication
3. Operator Addition
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra
using Printf

# Helper to generate 1D Hubbard model inputs
function generate_hubbard_1d(n_sites::Int, t::Float64, U::Float64)
    # Construct inputs
    hopping_io = IOBuffer()
    for i in 0:(n_sites-2)
        # Hopping between i and i+1
        # Use simple string concatenation instead of @printf into buffer for simplicity
        println(hopping_io, "$(2*i) $(2*(i+1)) $(-t) 0.0")
        println(hopping_io, "$(2*(i+1)) $(2*i) $(-t) 0.0")
        println(hopping_io, "$(2*i+1) $(2*(i+1)+1) $(-t) 0.0")
        println(hopping_io, "$(2*(i+1)+1) $(2*i+1) $(-t) 0.0")
    end
    # On-site energies
    for i in 0:(2*n_sites-1)
        println(hopping_io, "$i $i $(-U/2) 0.0")
    end

    coulomb_io = IOBuffer()
    println(coulomb_io, "$n_sites")
    for i in 0:(n_sites-1)
        # Coulomb U * n_up * n_down
        println(coulomb_io, "$i $(2*i) $(2*i+1) $(2*i+1) $(2*i) $U 0.0")
    end

    return String(take!(hopping_io)), String(take!(coulomb_io))
end

@testset "Performance Benchmark" begin

    # Shared Parameters
    t = 1.0
    U = 4.0

    @testset "Diagonalization Benchmark" begin
        # System: 4 sites (8 orbitals) -> 2^8 = 256 states
        # Small enough for fast Dense diag, good for comparison
        n_sites = 4
        n_flavors = 2 * n_sites
        dim = 2^n_flavors
        
        hop_str, coul_str = generate_hubbard_1d(n_sites, t, U)

        @printf("\n%s\n", "="^60)
        @printf("BENCHMARK: Diagonalization (Full Spectrum)\n")
        @printf("System: 1D Hubbard Chain (%d sites, %d orbitals)\n", n_sites, n_flavors)
        @printf("Hilbert Space Dimension: %d\n", dim)
        @printf("%s\n", "-"^60)
        @printf("%--15s | %-12s | %-12s\n", "Method", "Time (ms)", "Memory (MB)")
        @printf("%s\n", "-"^60)

        # --- Dense Method ---
        H_dense, _ = build_hamiltonian(IOBuffer(hop_str), IOBuffer(coul_str); method=:dense)
        
        # Warmup
        solve_hamiltonian(H_dense)
        
        # Benchmark
        stats_dense = @timed solve_hamiltonian(H_dense)
        t_dense = stats_dense.time * 1000
        m_dense = stats_dense.bytes / 1024^2
        
        @printf("%--15s | %12.4f | %12.2f\n", "Dense", t_dense, m_dense)

        # --- BlockDense Method ---
        site_qns = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:n_flavors]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hop_str), IOBuffer(coul_str); method=:blockdense, basis=basis)
        
        # Warmup
        solve_hamiltonian(H_block)
        
        # Benchmark
        stats_block = @timed solve_hamiltonian(H_block)
        t_block = stats_block.time * 1000
        m_block = stats_block.bytes / 1024^2
        
        @printf("%--15s | %12.4f | %12.2f\n", "BlockDense", t_block, m_block)
        @printf("%s\n", "-"^60)
        
        # Calculate comparison metrics
        speedup = t_dense / t_block
        mem_saving = m_dense / m_block
        
        # Theoretical speedup calculation
        sum_dim_cubed = sum(Float64(s.dim)^3 for s in basis.sectors)
        theory_speedup = (Float64(dim)^3) / sum_dim_cubed

        # Theoretical memory saving calculation
        sum_dim_squared = sum(Float64(s.dim)^2 for s in basis.sectors)
        theory_mem_saving = (Float64(dim)^2) / sum_dim_squared

        @printf("%--15s | %-12s | %-12s\n", "Metric", "Actual", "Theoretical")
        @printf("%s\n", "-"^60)
        @printf("%--15s | %11.2fx | %11.2fx\n", "Speedup", speedup, theory_speedup)
        @printf("%--15s | %11.2fx | %11.2fx\n", "Memory Saving", mem_saving, theory_mem_saving)
        @printf("%s\n", "="^60)
        
        @test speedup > 1.0
    end

    @testset "Operator Multiplication Benchmark" begin
        # System: 4 sites (8 orbitals) -> 2^8 = 256 states
        # Use matrix-matrix multiplication (H * H) as proxy for complex operations
        n_sites = 4
        n_flavors = 2 * n_sites
        dim = 2^n_flavors
        
        hop_str, coul_str = generate_hubbard_1d(n_sites, t, U)

        @printf("\n%s\n", "="^60)
        @printf("BENCHMARK: Operator Multiplication (H * H)\n")
        @printf("System: 1D Hubbard Chain (%d sites, %d orbitals)\n", n_sites, n_flavors)
        @printf("Hilbert Space Dimension: %d\n", dim)
        @printf("%s\n", "-"^60)
        @printf("%--15s | %-12s | %-12s\n", "Method", "Time (ms)", "Memory (MB)")
        @printf("%s\n", "-"^60)

        # --- Dense Method ---
        # Note: We construct Dense H but avoid full diagonalization as it might be slow
        # We focus on multiplication H * H
        H_dense, _ = build_hamiltonian(IOBuffer(hop_str), IOBuffer(coul_str); method=:dense)
        
        # Warmup
        _ = H_dense * H_dense
        
        # Benchmark
        stats_dense = @timed (H_dense * H_dense)
        t_dense = stats_dense.time * 1000
        m_dense = stats_dense.bytes / 1024^2
        
        @printf("%--15s | %12.4f | %12.2f\n", "Dense", t_dense, m_dense)

        # --- BlockDense Method ---
        site_qns = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:n_flavors]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hop_str), IOBuffer(coul_str); method=:blockdense, basis=basis)
        
        # Warmup
        _ = H_block * H_block
        
        # Benchmark
        stats_block = @timed (H_block * H_block)
        t_block = stats_block.time * 1000
        m_block = stats_block.bytes / 1024^2
        
        @printf("%--15s | %12.4f | %12.2f\n", "BlockDense", t_block, m_block)
        @printf("%s\n", "-"^60)
        
        # Calculate comparison metrics
        speedup = t_dense / t_block
        mem_saving = m_dense / m_block
        
        # Theoretical speedup calculation
        sum_dim_cubed = sum(Float64(s.dim)^3 for s in basis.sectors)
        theory_speedup = (Float64(dim)^3) / sum_dim_cubed

        # Theoretical memory saving calculation
        sum_dim_squared = sum(Float64(s.dim)^2 for s in basis.sectors)
        theory_mem_saving = (Float64(dim)^2) / sum_dim_squared

        @printf("%--15s | %-12s | %-12s\n", "Metric", "Actual", "Theoretical")
        @printf("%s\n", "-"^60)
        @printf("%--15s | %11.2fx | %11.2fx\n", "Speedup", speedup, theory_speedup)
        @printf("%--15s | %11.2fx | %11.2fx\n", "Memory Saving", mem_saving, theory_mem_saving)
        @printf("%s\n", "="^60)

        @test speedup > 1.0
    end

    @testset "Operator Addition Benchmark" begin
        # System: 4 sites (8 orbitals) -> 2^8 = 256 states
        # Test addition H + H
        n_sites = 4
        n_flavors = 2 * n_sites
        dim = 2^n_flavors
        
        hop_str, coul_str = generate_hubbard_1d(n_sites, t, U)

        @printf("\n%s\n", "="^60)
        @printf("BENCHMARK: Operator Addition (H + H)\n")
        @printf("System: 1D Hubbard Chain (%d sites, %d orbitals)\n", n_sites, n_flavors)
        @printf("Hilbert Space Dimension: %d\n", dim)
        @printf("%s\n", "-"^60)
        @printf("%--15s | %-12s | %-12s\n", "Method", "Time (ms)", "Memory (MB)")
        @printf("%s\n", "-"^60)

        # --- Dense Method ---
        H_dense, _ = build_hamiltonian(IOBuffer(hop_str), IOBuffer(coul_str); method=:dense)
        
        # Warmup
        _ = H_dense + H_dense
        
        # Benchmark
        stats_dense = @timed (H_dense + H_dense)
        t_dense = stats_dense.time * 1000
        m_dense = stats_dense.bytes / 1024^2
        
        @printf("%--15s | %12.4f | %12.2f\n", "Dense", t_dense, m_dense)

        # --- BlockDense Method ---
        site_qns = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:n_flavors]
        basis = make_basis(site_qns)
        H_block, _ = build_hamiltonian(IOBuffer(hop_str), IOBuffer(coul_str); method=:blockdense, basis=basis)
        
        # Test standard addition (+)
        stats_plus = @timed (H_block + H_block)
        t_plus = stats_plus.time * 1000
        m_plus = stats_plus.bytes / 1024^2
        
        # Test efficient sum()
        ops = [H_block, H_block]
        stats_sum = @timed sum(ops)
        t_sum = stats_sum.time * 1000
        m_sum = stats_sum.bytes / 1024^2
        
        @printf("%--15s | %12.4f | %12.2f\n", "BlockDense (+)", t_plus, m_plus)
        @printf("%--15s | %12.4f | %12.2f\n", "BlockDense sum", t_sum, m_sum)
        @printf("%s\n", "-"^60)
        
        # Comparison
        speedup = t_dense / t_plus
        mem_saving = m_dense / m_plus
        
        # Theoretical calculation for addition (O(N^2))
        sum_dim_squared = sum(Float64(s.dim)^2 for s in basis.sectors)
        theory_ratio = (Float64(dim)^2) / sum_dim_squared

        @printf("%--15s | %-12s | %-12s\n", "Metric", "Actual (+)", "Theoretical")
        @printf("%s\n", "-"^60)
        @printf("%--15s | %11.2fx | %11.2fx\n", "Speedup", speedup, theory_ratio)
        @printf("%--15s | %11.2fx | %11.2fx\n", "Memory Saving", mem_saving, theory_ratio)
        @printf("%s\n", "="^60)
        
        @printf("Optimization Gain (sum vs +): %.2fx speedup, %.2fx less memory\n", t_plus/t_sum, m_plus/m_sum)

        @test speedup > 1.0
    end

end
