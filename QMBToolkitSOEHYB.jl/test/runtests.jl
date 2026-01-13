using QMBToolkitSOEHYB
using Test
using HDF5
using MPI

# Initialize MPI once for all tests
MPI.Init()

# Shared test data directories
const DATA_DIR = joinpath(@__DIR__, "data")
const DATA_DIR_2SPIN = joinpath(@__DIR__, "data_2spin")

# Analytical values from analytic.nb (shared between Dense and BlockDense)
const Z_ANALYTICAL = [
    1.000000002061154,   # Z0 (order 0)
    3.501361871137541,   # Z0 + Z1 (order 1)
    7.114407091504815    # Z0 + Z1 + Z2 (order 2)
]

const G_ANALYTICAL = Dict(
    1 => [  # Order 1
        -0.999999997938846,
        -0.980198671286415,
        -0.960789437171989,
        -0.941764531643127
    ],
    2 => [  # Order 2
        -0.982110924312114,
        -0.962005704799589,
        -0.942394772571377,
        -0.923265608333401
    ],
    3 => [  # Order 3
        -0.966038824049318,
        -0.945639273769371,
        -0.925805877704756,
        -0.906522659615605
    ]
)

# Compute expected 2-spin partition function from single-spin values
# For two independent spins: Z_order_n = sum_{k=0}^{n} z_k * z_{n-k}
function compute_2spin_Z_analytical()
    # Extract per-order contributions from cumulative values
    z = zeros(3)
    z[1] = Z_ANALYTICAL[1]  # z0
    z[2] = Z_ANALYTICAL[2] - Z_ANALYTICAL[1]  # z1
    z[3] = Z_ANALYTICAL[3] - Z_ANALYTICAL[2]  # z2

    # Compute 2-spin cumulative partition function
    Z_2spin = zeros(3)
    Z_2spin[1] = z[1]^2  # order 0
    Z_2spin[2] = Z_2spin[1] + 2 * z[1] * z[2]  # cumulative up to order 1
    Z_2spin[3] = Z_2spin[2] + 2 * z[1] * z[3] + z[2]^2  # cumulative up to order 2

    return Z_2spin
end

const Z_ANALYTICAL_2SPIN = compute_2spin_Z_analytical()

"""
Compute expected 2-spin Green's function from single-spin analytical values.

For 2-spin with diagonal hybridization at order n:
Numerator^{(n)} = Σ_{j=0}^{n-1} g_{j+1} · z_{n-1-j}
G_2spin^{(n)} = Numerator^{(n)} / Z_2spin^{(n-1)}

where g_m = G_single^{(m)} * Z_single^{(m-1)} is the single-spin numerator.
"""
function compute_2spin_G_analytical()
    # Extract per-order partition function contributions
    z = zeros(3)
    z[1] = Z_ANALYTICAL[1]  # z0
    z[2] = Z_ANALYTICAL[2] - Z_ANALYTICAL[1]  # z1
    z[3] = Z_ANALYTICAL[3] - Z_ANALYTICAL[2]  # z2

    # Compute single-spin Green's function numerators: g_m = G^{(m)} * Z^{(m-1)}
    # G_ANALYTICAL has 4 tau points for each order
    n_tau = 4
    g = Dict{Int, Vector{Float64}}()
    for order in 1:3
        g[order] = G_ANALYTICAL[order] .* Z_ANALYTICAL[order]
    end

    # Compute 2-spin Green's function for each order
    G_2spin = Dict{Int, Vector{Float64}}()

    # Order 1: numerator = g_1 * z_0, denominator = Z_2spin^{(0)} = z_0^2
    # G_2spin^{(1)} = g_1 * z_0 / z_0^2 = g_1 / z_0 = G_single^{(1)}
    G_2spin[1] = (g[1] .* z[1]) ./ Z_ANALYTICAL_2SPIN[1]

    # Order 2: numerator = g_1 * z_1 + g_2 * z_0
    numerator_2 = g[1] .* z[2] .+ g[2] .* z[1]
    G_2spin[2] = numerator_2 ./ Z_ANALYTICAL_2SPIN[2]

    # Order 3: numerator = g_1 * z_2 + g_2 * z_1 + g_3 * z_0
    numerator_3 = g[1] .* z[3] .+ g[2] .* z[2] .+ g[3] .* z[1]
    G_2spin[3] = numerator_3 ./ Z_ANALYTICAL_2SPIN[3]

    return G_2spin
end

const G_ANALYTICAL_2SPIN = compute_2spin_G_analytical()

"""
Generate input.toml content for a given Hilbert space format and data directory.
"""
function generate_input_toml(format::String, data_dir::String = DATA_DIR)
    """
    [model]
    beta = 10.0
    file_path = "$(data_dir)"
    hopping_file = "hopping.txt"
    interaction_file = "Uijkl.txt"
    hybridization_poles_file = "hyb_poles.txt"
    hybridization_weights_file = "hyb_weights.txt"

    [solver]
    output_path = "."
    Hilbert_space_format = "$(format)"
    batch_size = 5000000

    [solver.propagator]
    is_bold = false
    expansion_order = 2

    [solver.green]
    is_bold = false
    expansion_order = 3
    save_linear_size = 1001

    [solver.dlr]
    Euv = 500.0
    rtol = 1e-14
    """
end

"""
Run bare expansion test for a given Hilbert space format.
"""
function run_bare_expansion_test(format::String)
    # Create temporary directory for this test
    test_dir = mktempdir()
    input_file = joinpath(test_dir, "input.toml")
    output_file = joinpath(test_dir, "sim.h5")

    # Write input file
    write(input_file, generate_input_toml(format))

    # Run solver
    QMBToolkitSOEHYB.run(input_file)

    # Check output file exists
    @test isfile(output_file)

    # Read results from HDF5
    h5open(output_file, "r") do file
        # Test partition function values
        Z_computed = read(file, "/partition_function/partition_function_by_order")

        @testset "Partition function - Order $order" for order in 0:2
            @test isapprox(Z_computed[order + 1], Z_ANALYTICAL[order + 1], rtol = 1e-12)
        end

        # Test Green's function values by order on linear grid
        tau_linear = read(file, "/green/linear_grid/tau")
        tau_test = [0.0, 0.01, 0.02, 0.03]

        @testset "Green's function - Order $order" for order in 1:3
            green_at_order = read(file, "/green/by_order/green_order_$order")

            for (tau_idx, tau_val) in enumerate(tau_test)
                linear_idx = argmin(abs.(tau_linear .- tau_val))
                G_computed = green_at_order[linear_idx, 1, 1]
                G_expected = G_ANALYTICAL[order][tau_idx]

                @test isapprox(G_computed, G_expected, rtol = 1e-12)
            end
        end
    end

    # Cleanup handled automatically by mktempdir()
end

"""
Run 2-spin bare expansion test for a given Hilbert space format.
Tests two independent, equivalent spins with diagonal hybridization.

The expected Green's function values are computed from single-spin analytical
data using the formula for independent spins (see compute_2spin_G_analytical).
"""
function run_2spin_bare_expansion_test(format::String)
    # Create temporary directory for this test
    test_dir = mktempdir()
    input_file = joinpath(test_dir, "input.toml")
    output_file = joinpath(test_dir, "sim.h5")

    # Write input file using 2-spin data
    write(input_file, generate_input_toml(format, DATA_DIR_2SPIN))

    # Run solver
    QMBToolkitSOEHYB.run(input_file)

    # Check output file exists
    @test isfile(output_file)

    # Read results from HDF5
    h5open(output_file, "r") do file
        # Test partition function values (2-spin expected values)
        Z_computed = read(file, "/partition_function/partition_function_by_order")

        @testset "Partition function - Order $order" for order in 0:2
            @test isapprox(Z_computed[order + 1], Z_ANALYTICAL_2SPIN[order + 1], rtol = 1e-12)
        end

        # Test Green's function values by order on linear grid
        # For independent equivalent spins with diagonal hybridization:
        # G[1,1] = G[2,2] = G_2spin (computed from single-spin data)
        # G[1,2] = G[2,1] = 0
        tau_linear = read(file, "/green/linear_grid/tau")
        tau_test = [0.0, 0.01, 0.02, 0.03]

        @testset "Green's function - Order $order" for order in 1:3
            green_at_order = read(file, "/green/by_order/green_order_$order")

            for (tau_idx, tau_val) in enumerate(tau_test)
                linear_idx = argmin(abs.(tau_linear .- tau_val))
                G_expected = G_ANALYTICAL_2SPIN[order][tau_idx]

                # Diagonal elements should match 2-spin analytical result
                G_00 = green_at_order[linear_idx, 1, 1]
                G_11 = green_at_order[linear_idx, 2, 2]
                @test isapprox(G_00, G_expected, rtol = 1e-12)
                @test isapprox(G_11, G_expected, rtol = 1e-12)

                # Off-diagonal elements should be zero
                G_01 = green_at_order[linear_idx, 1, 2]
                G_10 = green_at_order[linear_idx, 2, 1]
                @test isapprox(G_01, 0.0, atol = 1e-14)
                @test isapprox(G_10, 0.0, atol = 1e-14)
            end
        end
    end

    # Cleanup handled automatically by mktempdir()
end

@testset "QMBToolkitSOEHYB.jl" begin
    @testset "Bare expansion 1-spin ($format)" for format in ["dense", "blockdense"]
        run_bare_expansion_test(format)
    end
    @testset "Bare expansion 2-spin ($format)" for format in ["dense", "blockdense"]
        run_2spin_bare_expansion_test(format)
    end
end
