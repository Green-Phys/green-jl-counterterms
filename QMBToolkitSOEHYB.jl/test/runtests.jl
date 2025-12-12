using QMBToolkitSOEXCA
using Test
using HDF5
using MPI

# Initialize MPI once for all tests
MPI.Init()

@testset "QMBToolkitSOEXCA.jl" begin

    @testset "Bare expansion - Analytical comparison" begin
        # Test directory
        test_dir = joinpath(@__DIR__, "bare")
        input_file = joinpath(test_dir, "input.toml")
        output_file = joinpath(test_dir, "sim.h5")

        # Clean up any previous output
        if isfile(output_file)
            rm(output_file)
        end

        # Run solver
        QMBToolkitSOEXCA.run(input_file)

        # Check output file exists
        @test isfile(output_file)

        # Analytical partition function values from analytic.nb
        Z_analytical = [
            1.000000002061154,   # Z0 (order 0)
            3.501361871137541,   # Z0 + Z1 (order 1)
            7.114407091504815   # Z0 + Z1 + Z2 (order 2)
        ]

        # Analytical Green's function values at first 4 time points
        # Time points: τ = [0, 0.01, 0.02, 0.03] (from analytic.nb)
        # Green's function starts at order 1 (uses Z from propagator order 0)
        G_analytical = Dict(
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

        # Read results from HDF5
        h5open(output_file, "r") do file
            # Test partition function values
            Z_computed = read(file, "/partition_function/partition_function_by_order")

            @testset "Partition function - Order $order" for order in 0:2
                @test isapprox(Z_computed[order+1], Z_analytical[order+1], rtol=1e-12)
            end

            # Test Green's function values by order on linear grid
            tau_linear = read(file, "/green/linear_grid/tau")
            tau_test = [0.0, 0.01, 0.02, 0.03]

            @testset "Green's function - Order $order" for order in 1:3
                # Read Green's function at this order from linear grid
                green_at_order = read(file, "/green/by_order/green_order_$order")

                for (tau_idx, tau_val) in enumerate(tau_test)
                    # Find closest tau point in linear grid
                    linear_idx = argmin(abs.(tau_linear .- tau_val))

                    # Extract Green's function value (single orbital: [1,1] element)
                    G_computed = green_at_order[linear_idx, 1, 1]

                    # Compare with analytical result
                    G_expected = G_analytical[order][tau_idx]

                    @test isapprox(G_computed, G_expected, rtol=1e-12)
                end
            end
        end

        # Clean up
        rm(output_file)
    end

    @testset "Bare expansion (BlockDense) - Analytical comparison" begin
        # Test directory
        test_dir = joinpath(@__DIR__, "bare_blockdense")
        input_file = joinpath(test_dir, "input.toml")
        output_file = joinpath(test_dir, "sim.h5")

        # Clean up any previous output
        if isfile(output_file)
            rm(output_file)
        end

        # Run solver with BlockDense format
        QMBToolkitSOEXCA.run(input_file)

        # Check output file exists
        @test isfile(output_file)

        # Analytical partition function values (same as Dense case)
        Z_analytical = [
            1.000000002061154,   # Z0 (order 0)
            3.501361871137541,   # Z0 + Z1 (order 1)
            7.114407091504815   # Z0 + Z1 + Z2 (order 2)
        ]

        # Analytical Green's function values (same as Dense case)
        G_analytical = Dict(
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

        # Read results from HDF5
        h5open(output_file, "r") do file
            # Test partition function values
            Z_computed = read(file, "/partition_function/partition_function_by_order")

            @testset "Partition function (BlockDense) - Order $order" for order in 0:2
                @test isapprox(Z_computed[order+1], Z_analytical[order+1], rtol=1e-12)
            end

            # Test Green's function values by order on linear grid
            tau_linear = read(file, "/green/linear_grid/tau")
            tau_test = [0.0, 0.01, 0.02, 0.03]

            @testset "Green's function (BlockDense) - Order $order" for order in 1:3
                # Read Green's function at this order from linear grid
                green_at_order = read(file, "/green/by_order/green_order_$order")

                for (tau_idx, tau_val) in enumerate(tau_test)
                    # Find closest tau point in linear grid
                    linear_idx = argmin(abs.(tau_linear .- tau_val))

                    # Extract Green's function value (single orbital: [1,1] element)
                    G_computed = green_at_order[linear_idx, 1, 1]

                    # Compare with analytical result
                    G_expected = G_analytical[order][tau_idx]

                    @test isapprox(G_computed, G_expected, rtol=1e-12)
                end
            end
        end

        # Clean up
        rm(output_file)
    end
end
