"""
Performance Benchmark Tests

Compares Dense vs BlockDense performance on moderately-sized systems.
"""

using Test
using QMBToolkitED
using QMBToolkitED.BlockDense

@testset "Performance Benchmark" begin
    # 6 sites = 12 orbitals (spinful) -> 2^12 = 4096 states
    n_sites = 6
    n_flavors = 2 * n_sites
    t = 1.0
    U = 4.0

    # Construct inputs
    hopping_io = IOBuffer()
    for i in 0:(n_sites-2)
        # Hopping between i and i+1
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
        println(coulomb_io, "$i $(2*i) $(2*i+1) $(2*i+1) $(2*i) $U 0.0")
    end

    hopping_str = String(take!(hopping_io))
    coulomb_str = String(take!(coulomb_io))

    println("\n--- Performance Benchmark: 1D Hubbard Chain ($n_sites sites, $n_flavors orbitals) ---")
    println("Hilbert Space Dimension: $(2^n_flavors)")

    # Theoretical estimate
    site_qns = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:n_flavors]
    basis = make_basis(site_qns)

    total_dim = 2^n_flavors
    sum_dim_cubed = sum(Float64(s.dim)^3 for s in basis.sectors)
    theory_speedup = (Float64(total_dim)^3) / sum_dim_cubed
    println("Theoretical Speedup (sum d_i^3 vs D^3): $(round(theory_speedup, digits=2))x")

    # BlockDense timing
    H_block, _ = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str); method=:blockdense, basis=basis)

    println("Running BlockDense diagonalization...")
    solve_hamiltonian(H_block)  # Compile run

    stats_block = @timed solve_hamiltonian(H_block)
    time_block = stats_block.time
    mem_block = stats_block.bytes / 1024^2

    println("BlockDense: $(round(time_block, digits=4)) s, $(round(mem_block, digits=2)) MB")

    # Comparative benchmark with smaller system
    n_sites_comp = 4
    n_flavors_comp = 2 * n_sites_comp
    println("\n--- Comparative Benchmark: 1D Hubbard Chain ($n_sites_comp sites, $n_flavors_comp orbitals) ---")
    println("Hilbert Space Dimension: $(2^n_flavors_comp) ($(2^n_flavors_comp))")

    # Generate inputs for smaller system
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
        println(coulomb_io_comp, "$i $(2*i) $(2*i+1) $(2*i+1) $(2*i) $U 0.0")
    end

    hop_str_c = String(take!(hopping_io_comp))
    coul_str_c = String(take!(coulomb_io_comp))

    # Dense run
    H_dense_c, _ = build_hamiltonian(IOBuffer(hop_str_c), IOBuffer(coul_str_c))
    println("Running Dense diagonalization...")
    solve_hamiltonian(H_dense_c)  # Compile
    stats_dense = @timed solve_hamiltonian(H_dense_c)

    # BlockDense run
    site_qns_c = [QN_NSz(1, isodd(i) ? 1 : -1) for i in 1:n_flavors_comp]
    basis_c = make_basis(site_qns_c)
    H_block_c, _ = build_hamiltonian(IOBuffer(hop_str_c), IOBuffer(coul_str_c); method=:blockdense, basis=basis_c)
    println("Running BlockDense diagonalization...")
    solve_hamiltonian(H_block_c)  # Compile
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

    # Theoretical speedup for comparative benchmark
    total_dim_c = 2^n_flavors_comp
    sum_dim_cubed_c = sum(Float64(s.dim)^3 for s in basis_c.sectors)
    theory_speedup_comp = (Float64(total_dim_c)^3) / sum_dim_cubed_c
    println("Theoretical Speedup (N=$(n_flavors_comp)): $(round(theory_speedup_comp, digits=2))x")

    @test speedup > 1.0  # Should be faster
end
