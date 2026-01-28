"""
Benchmarks: Operator (block-sparse) vs Dense Matrix operations.

This script compares the performance of the block-sparse Operator type against
equivalent dense matrix operations across different system sizes and quantum
number types.

Usage:
    julia --project test/benchmarks/run_benchmarks.jl [nflavors_max]

    nflavors_max: Maximum number of flavors to benchmark (default: 10).
                  Benchmarks run for nflavors = 4, 6, 8, ..., nflavors_max.

Examples:
    julia --project test/benchmarks/run_benchmarks.jl
    julia --project test/benchmarks/run_benchmarks.jl 12
"""

using QMBToolkitED
using LinearAlgebra

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Warm up a function by calling it once (trigger JIT compilation)."""
macro warmup(expr)
    quote
        $(esc(expr))
        nothing
    end
end

"""Time an expression over `nruns` repetitions and return the minimum elapsed time."""
macro bench(nruns, expr)
    quote
        local best = Inf
        for _ in 1:$(esc(nruns))
            local t = @elapsed $(esc(expr))
            best = min(best, t)
        end
        best
    end
end

function separator(char='=', width=80)
    println(repeat(char, width))
end

function header(title::String)
    separator()
    println("  ", title)
    separator()
end

function format_time(t::Float64)
    if t < 1e-6
        return @sprintf("%.2f ns", t * 1e9)
    elseif t < 1e-3
        return @sprintf("%.2f μs", t * 1e6)
    elseif t < 1.0
        return @sprintf("%.2f ms", t * 1e3)
    else
        return @sprintf("%.2f s", t)
    end
end

using Printf

function print_row(label::String, t_op::Float64, t_mat::Float64)
    ratio = t_mat / t_op
    @printf("  %-30s  %12s  %12s  %8.2fx\n",
            label, format_time(t_op), format_time(t_mat), ratio)
end

function print_header_row()
    @printf("  %-30s  %12s  %12s  %8s\n",
            "Operation", "Operator", "Dense", "Speedup")
    println("  ", repeat('-', 68))
end

# ---------------------------------------------------------------------------
# Benchmark suite for a given quantum number type and system size
# ---------------------------------------------------------------------------

function run_suite(qn_label::String, flavor_qns::Vector{<:QuantumNumber}, nruns::Int)
    nf = length(flavor_qns)
    dim = 2^nf
    n_sectors = 0

    println()
    println("  QN type: $qn_label | nflavors: $nf | Hilbert dim: $dim")
    separator('-', 80)

    # --- Setup ---
    basis = Basis(flavor_qns)
    n_sectors = length(basis.sectors)
    println("  Sectors: $n_sectors | Largest sector dim: ",
            maximum(s.dim for s in basis.sectors))
    println()

    c_ops = annihilators(basis)
    c_mats = [to_matrix(op) for op in c_ops]

    print_header_row()

    # --- 1. Multiplication: c†_1 * c_2 ---
    @warmup c_ops[1]' * c_ops[min(2, nf)]
    @warmup c_mats[1]' * c_mats[min(2, nf)]

    t_op  = @bench nruns c_ops[1]' * c_ops[min(2, nf)]
    t_mat = @bench nruns c_mats[1]' * c_mats[min(2, nf)]
    print_row("Multiply (c†₁ * c₂)", t_op, t_mat)

    # --- 2. Addition: c_1 + c_2 ---
    if nf >= 2
        @warmup c_ops[1] + c_ops[2]
        @warmup c_mats[1] + c_mats[2]

        t_op  = @bench nruns c_ops[1] + c_ops[2]
        t_mat = @bench nruns c_mats[1] + c_mats[2]
        print_row("Addition (c₁ + c₂)", t_op, t_mat)
    end

    # --- 3. Adjoint ---
    @warmup c_ops[1]'
    @warmup c_mats[1]'

    t_op  = @bench nruns c_ops[1]'
    t_mat = @bench nruns c_mats[1]'
    print_row("Adjoint (c₁†)", t_op, t_mat)

    # --- 4. Scalar multiplication ---
    @warmup 2.5 * c_ops[1]
    @warmup 2.5 * c_mats[1]

    t_op  = @bench nruns 2.5 * c_ops[1]
    t_mat = @bench nruns 2.5 * c_mats[1]
    print_row("Scalar multiply (α * c₁)", t_op, t_mat)

    # --- 5. Number operator: n_total = Σ c†_i c_i ---
    function build_n_op(c)
        terms = [c[i]' * c[i] for i in 1:length(c)]
        return sum(terms)
    end
    function build_n_mat(cm)
        terms = [cm[i]' * cm[i] for i in 1:length(cm)]
        return sum(terms)
    end

    @warmup build_n_op(c_ops)
    @warmup build_n_mat(c_mats)

    t_op  = @bench nruns build_n_op(c_ops)
    t_mat = @bench nruns build_n_mat(c_mats)
    print_row("Number op (Σ c†ᵢcᵢ)", t_op, t_mat)

    # --- 6. Hopping Hamiltonian: H = Σ_{i≠j} c†_i c_j ---
    function build_hop_op(c)
        terms = Operator[]
        for i in 1:length(c), j in 1:length(c)
            if i != j
                push!(terms, c[i]' * c[j])
            end
        end
        return sum(terms)
    end
    function build_hop_mat(cm)
        acc = zeros(ComplexF64, size(cm[1]))
        for i in 1:length(cm), j in 1:length(cm)
            if i != j
                acc .+= cm[i]' * cm[j]
            end
        end
        return acc
    end

    @warmup build_hop_op(c_ops)
    @warmup build_hop_mat(c_mats)

    t_op  = @bench nruns build_hop_op(c_ops)
    t_mat = @bench nruns build_hop_mat(c_mats)
    print_row("Hopping H (Σ c†ᵢcⱼ)", t_op, t_mat)

    # --- 7. Diagonalization ---
    H_op = build_hop_op(c_ops) + build_n_op(c_ops)
    H_mat = to_matrix(H_op)

    @warmup solve_hamiltonian(H_op)
    @warmup eigen(Hermitian(H_mat))

    t_op  = @bench nruns solve_hamiltonian(H_op)
    t_mat = @bench nruns eigen(Hermitian(H_mat))
    print_row("Diagonalization", t_op, t_mat)

    # --- 8. Conversion to_matrix overhead ---
    @warmup to_matrix(H_op)

    t_conv = @bench nruns to_matrix(H_op)
    @printf("  %-30s  %12s\n", "to_matrix(H) overhead", format_time(t_conv))

    println()
    return n_sectors
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    nflavors_max = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    nflavors_list = 4:2:nflavors_max

    if isempty(nflavors_list)
        println("No valid nflavors to benchmark. Provide nflavors_max >= 4.")
        return
    end

    header("QMBToolkitED Operator vs Dense Matrix Benchmarks")
    println()
    println("  System sizes (nflavors): ", collect(nflavors_list))
    println("  QN types: QN_Trivial, QN_N, QN_NSz")
    println()

    # Determine number of repetitions based on max size
    # Larger systems need fewer repetitions since each run is slower
    nruns_map = Dict(
        4  => 100,
        6  => 50,
        8  => 20,
        10 => 5,
        12 => 3,
        14 => 1,
    )

    for nf in nflavors_list
        nruns = get(nruns_map, nf, 1)

        header("nflavors = $nf (Hilbert space dim = $(2^nf), nruns = $nruns)")

        # QN_Trivial: single sector, no block-sparse advantage
        flavor_qns_trivial = [QN_Trivial() for _ in 1:nf]
        run_suite("QN_Trivial", flavor_qns_trivial, nruns)

        # QN_N: nf+1 sectors
        flavor_qns_n = [QN_N(1) for _ in 1:nf]
        run_suite("QN_N", flavor_qns_n, nruns)

        # QN_NSz: more sectors, smaller blocks
        # Half spin-up, half spin-down
        flavor_qns_nsz = vcat(
            [QN_NSz(1, 1) for _ in 1:nf÷2],
            [QN_NSz(1, -1) for _ in 1:nf÷2]
        )
        run_suite("QN_NSz", flavor_qns_nsz, nruns)
    end

    header("Benchmark Complete")
end

main()
