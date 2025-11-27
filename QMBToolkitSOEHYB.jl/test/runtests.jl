using QMBToolkitSOEXCA
using Test
using HDF5

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
        G_analytical = Dict(
            0 => [  # Order 0
                -0.999999997938846,
                -0.980198671286415,
                -0.960789437171989,
                -0.941764531643127
            ],
            1 => [  # Order 1
                -0.982110924312114,
                -0.962005704799589,
                -0.942394772571377,
                -0.923265608333401
            ],
            2 => [  # Order 2
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
                @test isapprox(Z_computed[order+1], Z_analytical[order+1], rtol=1e-14)
            end

            # Test Green's function values
            # Read Green's function on linear grid
            green_linear = read(file, "/green/linear_grid/green_tau")
            tau_linear = read(file, "/green/linear_grid/tau")

            # Find indices for the first 4 time points: τ = [0, 0.01, 0.02, 0.03]
            tau_test = [0.0, 0.01, 0.02, 0.03]

            @testset "Green's function" begin
                # For this test case: single orbital (1x1 matrix at each tau point)
                # Green's function should be stored as green_linear[tau_idx, flavor_i, flavor_j]
                # For single orbital: flavor_i = flavor_j = 1

                for (tau_idx, tau_val) in enumerate(tau_test)
                    # Find closest tau point in linear grid
                    linear_idx = argmin(abs.(tau_linear .- tau_val))

                    # Extract Green's function value (single orbital: [1,1] element)
                    G_computed = green_linear[linear_idx, 1, 1]

                    # Compare with order 2 analytical result (highest available order)
                    G_expected = G_analytical[2][tau_idx]

                    @test isapprox(G_computed, G_expected, rtol=1e-14)
                end
            end
        end

        # Clean up
        rm(output_file)
    end
end
