"""
Benchmarks: Operator (block-sparse) vs Dense Matrix basic operations.

Compares speed and memory of block-sparse Operator arithmetic against equivalent
dense matrix operations, and validates measured ratios against theoretical
expectations derived from the block structure.

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
using Printf

# ============================================================================
# Measurement helpers
# ============================================================================

"""Minimum elapsed time over `nruns` repetitions (triggers JIT on first call)."""
macro bench_time(nruns, expr)
    quote
        local best = Inf
        for _ in 1:$(esc(nruns))
            local t = @elapsed $(esc(expr))
            best = min(best, t)
        end
        best
    end
end

"""Bytes allocated for a single evaluation (after one warmup call)."""
macro bench_alloc(expr)
    quote
        $(esc(expr))                    # warmup
        @allocated $(esc(expr))         # measure
    end
end

# ============================================================================
# Theoretical ratio computation
# ============================================================================

"""Total number of ComplexF64 elements stored across all blocks of an Operator."""
function block_elements(op::Operator)
    total = 0
    for block in values(op.blocks)
        total += length(block)
    end
    return total
end

"""
Theoretical memory ratio for storing the *result* of an element-wise operation
(addition, scalar multiply, adjoint).

For the result operator with block sizes {d_tgt × d_src}, the ratio is:
    N² / Σ_blocks (d_tgt × d_src)
where N = 2^nflavors is the full Hilbert space dimension.
"""
function theory_memory_elementwise(result_op::Operator)
    N = result_op.basis.total_dim
    elems = block_elements(result_op)
    return elems == 0 ? 1.0 : N^2 / elems
end

"""
Theoretical speed ratio for element-wise operations (add, scalar mul, adjoint).

These are O(n²) in the number of matrix elements, so the ratio equals the
memory ratio:
    N² / Σ_blocks (d_tgt × d_src)
"""
theory_speed_elementwise(result_op::Operator) = theory_memory_elementwise(result_op)

"""
Theoretical speed ratio for multiplication A * B.

Dense multiplication: O(N³)
Block-sparse: O(Σ_{valid chains} d_tgt × d_mid × d_src)
  where a "chain" is a triple (tgt, mid, src) with A[tgt,mid] and B[mid,src] both present.

Returns N³ / total_block_flops.
"""
function theory_speed_multiply(A::Operator, B::Operator)
    N = A.basis.total_dim

    # Index B blocks by their target sector (= the intermediate sector "mid")
    B_by_mid = Dict{Int, Vector{Tuple{Int, Int, Int}}}()  # mid -> [(src, rows, cols)]
    for ((mid, src), block) in B.blocks
        entry = (src, size(block, 1), size(block, 2))
        if haskey(B_by_mid, mid)
            push!(B_by_mid[mid], entry)
        else
            B_by_mid[mid] = [entry]
        end
    end

    total_flops = 0
    for ((tgt, mid), A_block) in A.blocks
        d_tgt = size(A_block, 1)
        d_mid = size(A_block, 2)
        if haskey(B_by_mid, mid)
            for (_, _, d_src) in B_by_mid[mid]
                total_flops += d_tgt * d_mid * d_src
            end
        end
    end

    return total_flops == 0 ? 1.0 : N^3 / total_flops
end

"""
Theoretical memory ratio for multiplication result.

Dense result: N² elements.
Block-sparse result: Σ (d_tgt × d_src) over result blocks.
"""
function theory_memory_multiply(result_op::Operator)
    return theory_memory_elementwise(result_op)
end

# ============================================================================
# Formatting
# ============================================================================

function format_time(t::Float64)
    if t < 1e-6
        @sprintf("%.1f ns", t * 1e9)
    elseif t < 1e-3
        @sprintf("%.1f μs", t * 1e6)
    elseif t < 1.0
        @sprintf("%.2f ms", t * 1e3)
    else
        @sprintf("%.2f s", t)
    end
end

function format_bytes(b::Int)
    if b < 1024
        @sprintf("%d B", b)
    elseif b < 1024^2
        @sprintf("%.1f KB", b / 1024)
    elseif b < 1024^3
        @sprintf("%.1f MB", b / 1024^2)
    else
        @sprintf("%.1f GB", b / 1024^3)
    end
end

function separator(char='=', width=82)
    println(repeat(char, width))
end

function print_op_header(name::String)
    println()
    @printf("  %-40s %10s %10s %9s %9s\n", name, "Operator", "Dense", "Ratio", "Theory")
    println("  ", repeat('-', 78))
end

function print_metric(label::String, val_op, val_dense, measured_ratio::Float64, theory::Float64;
                      format_fn=format_time)
    @printf("    %-38s %10s %10s %8.1fx %8.1fx\n",
            label, format_fn(val_op), format_fn(val_dense), measured_ratio, theory)
end

# ============================================================================
# Benchmark runner for one (nflavors, QN_type) combination
# ============================================================================

function run_suite(qn_label::String, flavor_qns::Vector{<:QuantumNumber}, nruns::Int)
    nf = length(flavor_qns)
    dim = 2^nf

    basis = Basis(flavor_qns)
    nsectors = length(basis.sectors)
    max_sector_dim = maximum(s.dim for s in basis.sectors)

    println()
    @printf("  QN type: %-12s  nflavors: %d  dim: %d  sectors: %d  max block: %d\n",
            qn_label, nf, dim, nsectors, max_sector_dim)
    separator('-', 82)

    c_ops = annihilators(basis)
    c_mats = [to_matrix(op) for op in c_ops]
    j = min(2, nf)  # second flavor index

    # ------------------------------------------------------------------
    # 1. Multiplication: c†₁ * c₂
    # ------------------------------------------------------------------
    op_a = c_ops[1]'
    op_b = c_ops[j]
    mat_a = c_mats[1]'
    mat_b = c_mats[j]

    result_op = op_a * op_b
    result_mat = mat_a * mat_b

    t_op  = @bench_time nruns (op_a * op_b)
    t_mat = @bench_time nruns (mat_a * mat_b)
    m_op  = @bench_alloc (op_a * op_b)
    m_mat = @bench_alloc (mat_a * mat_b)

    theory_t = theory_speed_multiply(op_a, op_b)
    theory_m = theory_memory_multiply(result_op)

    print_op_header("Multiplication (c†₁ * c₂)")
    print_metric("Time",   t_op, t_mat, t_mat/t_op, theory_t)
    print_metric("Memory", m_op, m_mat, m_mat/max(m_op,1), theory_m; format_fn=format_bytes)

    # ------------------------------------------------------------------
    # 2. Addition: c₁ + c₂  (same flux)
    # ------------------------------------------------------------------
    if nf >= 2
        result_op = c_ops[1] + c_ops[2]

        t_op  = @bench_time nruns (c_ops[1] + c_ops[2])
        t_mat = @bench_time nruns (c_mats[1] + c_mats[2])
        m_op  = @bench_alloc (c_ops[1] + c_ops[2])
        m_mat = @bench_alloc (c_mats[1] + c_mats[2])

        theory_t = theory_speed_elementwise(result_op)
        theory_m = theory_memory_elementwise(result_op)

        print_op_header("Addition (c₁ + c₂)")
        print_metric("Time",   t_op, t_mat, t_mat/t_op, theory_t)
        print_metric("Memory", m_op, m_mat, m_mat/max(m_op,1), theory_m; format_fn=format_bytes)
    end

    # ------------------------------------------------------------------
    # 3. Adjoint: c₁†
    #    Note: Julia's adjoint on a dense matrix returns a lazy wrapper (zero cost).
    #    We use collect(M') to force materialization for a fair comparison.
    # ------------------------------------------------------------------
    result_op = c_ops[1]'

    t_op  = @bench_time nruns c_ops[1]'
    t_mat = @bench_time nruns collect(c_mats[1]')
    m_op  = @bench_alloc c_ops[1]'
    m_mat = @bench_alloc collect(c_mats[1]')

    theory_t = theory_speed_elementwise(result_op)
    theory_m = theory_memory_elementwise(result_op)

    print_op_header("Adjoint (c₁†)")
    print_metric("Time",   t_op, t_mat, t_mat/t_op, theory_t)
    print_metric("Memory", m_op, m_mat, m_mat/max(m_op,1), theory_m; format_fn=format_bytes)

    # ------------------------------------------------------------------
    # 4. Scalar multiplication: α * c₁
    # ------------------------------------------------------------------
    result_op = 2.5 * c_ops[1]

    t_op  = @bench_time nruns (2.5 * c_ops[1])
    t_mat = @bench_time nruns (2.5 * c_mats[1])
    m_op  = @bench_alloc (2.5 * c_ops[1])
    m_mat = @bench_alloc (2.5 * c_mats[1])

    theory_t = theory_speed_elementwise(result_op)
    theory_m = theory_memory_elementwise(result_op)

    print_op_header("Scalar multiply (α * c₁)")
    print_metric("Time",   t_op, t_mat, t_mat/t_op, theory_t)
    print_metric("Memory", m_op, m_mat, m_mat/max(m_op,1), theory_m; format_fn=format_bytes)

    println()
end

# ============================================================================
# Main
# ============================================================================

function main()
    nflavors_max = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    nflavors_list = 4:2:nflavors_max

    if isempty(nflavors_list)
        println("No valid nflavors to benchmark. Provide nflavors_max >= 4.")
        return
    end

    separator()
    println("  QMBToolkitED: Operator vs Dense Matrix — Basic Operations")
    separator()
    println()
    println("  System sizes (nflavors): ", collect(nflavors_list))
    println("  QN types: QN_Trivial, QN_N, QN_NSz")
    println()
    println("  For each operation we report:")
    println("    Ratio  = Dense / Operator  (measured speedup or memory saving)")
    println("    Theory = expected ratio from block structure analysis")
    println()
    println("  Theoretical ratios:")
    println("    Element-wise ops (add, scalar, adjoint): N² / Σ_blocks(dᵢ × dⱼ)")
    println("    Multiplication:  N³ / Σ_chains(d_tgt × d_mid × d_src)")
    println("    Memory:          N² / Σ_result_blocks(dᵢ × dⱼ)")
    println()

    nruns_map = Dict(4 => 200, 6 => 100, 8 => 30, 10 => 10, 12 => 3, 14 => 1)

    for nf in nflavors_list
        nruns = get(nruns_map, nf, 1)

        separator()
        @printf("  nflavors = %d    Hilbert dim = %d    repetitions = %d\n", nf, 2^nf, nruns)
        separator()

        # QN_Trivial: 1 sector, no block advantage (baseline)
        run_suite("QN_Trivial", [QN_Trivial() for _ in 1:nf], nruns)

        # QN_N: nf+1 sectors
        run_suite("QN_N", [QN_N(1) for _ in 1:nf], nruns)

        # QN_NSz: most sectors, smallest blocks
        flavor_qns_nsz = vcat(
            [QN_NSz(1, 1)  for _ in 1:nf÷2],
            [QN_NSz(1, -1) for _ in 1:nf÷2]
        )
        run_suite("QN_NSz", flavor_qns_nsz, nruns)
    end

    separator()
    println("  Benchmark complete.")
    separator()
end

main()
