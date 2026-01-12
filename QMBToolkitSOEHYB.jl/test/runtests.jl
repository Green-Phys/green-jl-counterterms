using QMBToolkitSOEHYB
using Test
using HDF5
using MPI

# Initialize MPI once for all tests
MPI.Init()

# Shared test data directory
const DATA_DIR = joinpath(@__DIR__, "data")

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

"""
Generate input.toml content for a given Hilbert space format.
"""
function generate_input_toml(format::String)
    """
    [model]
    beta = 10.0
    file_path = "$(DATA_DIR)"
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

@testset "QMBToolkitSOEHYB.jl" begin
    @testset "Bare expansion ($format)" for format in ["dense", "blockdense"]
        run_bare_expansion_test(format)
    end
end
