using Test
using QMBToolkitDLR
using QMBToolkitED
using QMBToolkitED.BlockDense
using LinearAlgebra
using Printf

@testset "BlockDense DLR Operations" begin
    # Setup: Create a simple 2-site system with QN_N conservation
    # This gives us 3 sectors: N=0 (dim=1), N=1 (dim=2), N=2 (dim=1)
    site_qns = [QN_N(1), QN_N(1)]
    basis = make_basis(site_qns)

    beta = 1000.0
    Euv = 1000.0
    rtol = 1e-12
    dlr_grid = DLRGrid(Euv, beta, rtol, true)
    r = dlr_grid.size
    n_tau = length(dlr_grid.tau)
    tau_grid = dlr_grid.tau

    println("\n" * "="^40)
    println("BlockDense DLR Test Setup")
    println("="^40)
    println("Basis sectors:")
    for (i, sector) in enumerate(basis.sectors)
        println("  Sector $i: qn=$(sector.qn), dim=$(sector.dim)")
    end
    println("DLR grid: rank r = $r, n_tau = $n_tau")

    @testset "matrix convolution BlockDense" begin
        # Use same frequencies as dense test
        omega_F = 0.1234
        omega_G = -0.5678

        # Coefficient matrices (will be used in each block)
        F_coeffs = [1.0 2.0; 3.0 4.0]
        G_coeffs = [5.0 6.0; 7.0 8.0]

        # Create BlockDenseArrays with block-diagonal structure
        # Each diagonal block gets the same matrix structure scaled by kernel
        flux_zero = OperatorFlux(zero_qn(QN_N))

        # Build F: each block is kernel_F(τ) * F_coeffs
        F_blocks_tau = Dict{Tuple{Int,Int}, Array{ComplexF64,3}}()
        for (sector_idx, sector) in enumerate(basis.sectors)
            dim = sector.dim
            if dim == 2  # Only N=1 sector has dim=2, perfect for 2x2 matrix test
                block = zeros(ComplexF64, n_tau, dim, dim)
                for k in 1:n_tau
                    tau = tau_grid[k]
                    block[k, :, :] = kernel_tau(tau, omega_F, beta) * F_coeffs
                end
                F_blocks_tau[(sector_idx, sector_idx)] = block
            else  # For other sectors (dim=1), use scalar
                block = zeros(ComplexF64, n_tau, dim, dim)
                for k in 1:n_tau
                    tau = tau_grid[k]
                    block[k, 1, 1] = kernel_tau(tau, omega_F, beta)
                end
                F_blocks_tau[(sector_idx, sector_idx)] = block
            end
        end

        # Build G: each block is kernel_G(τ) * G_coeffs
        G_blocks_tau = Dict{Tuple{Int,Int}, Array{ComplexF64,3}}()
        for (sector_idx, sector) in enumerate(basis.sectors)
            dim = sector.dim
            if dim == 2
                block = zeros(ComplexF64, n_tau, dim, dim)
                for k in 1:n_tau
                    tau = tau_grid[k]
                    block[k, :, :] = kernel_tau(tau, omega_G, beta) * G_coeffs
                end
                G_blocks_tau[(sector_idx, sector_idx)] = block
            else
                block = zeros(ComplexF64, n_tau, dim, dim)
                for k in 1:n_tau
                    tau = tau_grid[k]
                    block[k, 1, 1] = kernel_tau(tau, omega_G, beta)
                end
                G_blocks_tau[(sector_idx, sector_idx)] = block
            end
        end

        F_tau = BlockDenseArray{ComplexF64}(basis, flux_zero, F_blocks_tau)
        G_tau = BlockDenseArray{ComplexF64}(basis, flux_zero, G_blocks_tau)

        # Transform to DLR
        F_dlr = tau2dlr_blockdense(dlr_grid, F_tau, tau_grid)
        G_dlr = tau2dlr_blockdense(dlr_grid, G_tau, tau_grid)

        # Create convolution workspace
        kernel_cache = compute_kernel_cache(dlr_grid)
        workspace = create_blockdense_convolution_workspace(basis, flux_zero, flux_zero, r, n_tau, ComplexF64)

        # Compute convolution H = F * G
        H_tau = do_convolution_dlr2tau_blockdense!(workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)
        H_dlr = tau2dlr_blockdense(dlr_grid, H_tau, tau_grid)

        # Test on fine grid
        tau_grid_test = collect(LinRange(0.0, beta, 10000))
        H_reconstructed = dlr2tau_blockdense(dlr_grid, H_dlr, tau_grid_test)

        # Analytical result: h(τ) = [f(τ)*g(0) - f(0)*g(τ)] / (ω_F - ω_G)
        h_analytical_scalar = [(kernel_tau(tau, omega_G, beta)*kernel_tau(0.0, omega_F, beta) -
                                kernel_tau(tau, omega_F, beta)*kernel_tau(0.0, omega_G, beta)) / (omega_F - omega_G)
                               for tau in tau_grid_test]

        # For N=1 sector (dim=2): H = h(τ) * (F_coeffs * G_coeffs)
        H_coeffs = F_coeffs * G_coeffs

        # ===== Test scalar convolution (dim=1 sectors) =====
        max_error_scalar = 0.0
        sum_sq_error_scalar = 0.0
        n_elements_scalar = 0
        max_val_scalar = 0.0
        max_tau_idx_scalar = 1

        for sector_idx in [1, 3]  # N=0 and N=2 sectors (both have dim=1)
            H_block_scalar = H_reconstructed.blocks[(sector_idx, sector_idx)]
            for k in 1:length(tau_grid_test)
                expected_scalar = h_analytical_scalar[k]
                actual_scalar = H_block_scalar[k, 1, 1]
                error = abs(actual_scalar - expected_scalar)
                max_error_scalar = max(max_error_scalar, error)
                sum_sq_error_scalar += error^2
                n_elements_scalar += 1

                if abs(expected_scalar) > abs(max_val_scalar)
                    max_val_scalar = expected_scalar
                    max_tau_idx_scalar = k
                end
            end
        end

        err_inf_scalar = max_error_scalar
        err_l2_scalar = sqrt(sum_sq_error_scalar / n_elements_scalar)

        # Get actual value at max location for scalar
        H_block_scalar_check = H_reconstructed.blocks[(1, 1)]
        actual_at_max_scalar = H_block_scalar_check[max_tau_idx_scalar, 1, 1]

        println("="^40)
        println("Scalar Convolution BlockDense Test")
        @printf("Max value of H analytical: %.3e at tau=%.3e with error %.3e\n", max_val_scalar, tau_grid_test[max_tau_idx_scalar], abs(actual_at_max_scalar - max_val_scalar))
        @printf("Scalar convolution BlockDense test: ||error||_inf = %.3e, ||error||_2 = %.3e\n", err_inf_scalar, err_l2_scalar)

        # ===== Test matrix convolution (dim=2 sector) =====
        sector_idx_matrix = 2
        H_block_reconstructed = H_reconstructed.blocks[(sector_idx_matrix, sector_idx_matrix)]

        # Compute errors for matrix block
        max_error_matrix = 0.0
        sum_sq_error_matrix = 0.0
        n_elements_matrix = 0
        max_val_matrix = 0.0
        max_tau_idx_matrix = 1
        max_i, max_j = 1, 1

        for k in 1:length(tau_grid_test)
            expected = h_analytical_scalar[k] * H_coeffs
            actual = H_block_reconstructed[k, :, :]
            error = maximum(abs.(actual - expected))
            max_error_matrix = max(max_error_matrix, error)
            sum_sq_error_matrix += sum(abs.(actual - expected).^2)
            n_elements_matrix += length(expected)

            for i in 1:2, j in 1:2
                if abs(expected[i,j]) > abs(max_val_matrix)
                    max_val_matrix = expected[i,j]
                    max_tau_idx_matrix = k
                    max_i, max_j = i, j
                end
            end
        end

        err_inf_matrix = max_error_matrix
        err_l2_matrix = sqrt(sum_sq_error_matrix / n_elements_matrix)
        actual_at_max_matrix = H_block_reconstructed[max_tau_idx_matrix, max_i, max_j]

        println("="^40)
        println("Matrix Convolution BlockDense Test")
        @printf("Max value of H analytical: %.3e at tau=%.3e with error %.3e\n", max_val_matrix, tau_grid_test[max_tau_idx_matrix], abs(actual_at_max_matrix - max_val_matrix))
        @printf("Matrix Convolution BlockDense test: ||error||_inf = %.3e, ||error||_2 = %.3e\n", err_inf_matrix, err_l2_matrix)

        # Tests pass if both scalar and matrix are correct
        @test err_inf_scalar < rtol*100
        @test err_l2_scalar < rtol*100
        @test err_inf_matrix < rtol*100 * maximum(abs.(F_coeffs)) * maximum(abs.(G_coeffs))
        @test err_l2_matrix < rtol*100 * maximum(abs.(F_coeffs)) * maximum(abs.(G_coeffs))
    end

    @testset "reflection BlockDense" begin

        beta_refl = 10.0
        Euv_refl = 10.0
        dlr_grid_refl = DLRGrid(Euv_refl, beta_refl, rtol, true)
        tau_grid_refl = dlr_grid_refl.tau

        # Omega matrix for different blocks
        omega_matrix = [0.1234 0.5678; -0.9012 -0.3456]

        flux_zero = OperatorFlux(zero_qn(QN_N))
        H_blocks_tau = Dict{Tuple{Int,Int}, Array{Float64,3}}()

        for (sector_idx, sector) in enumerate(basis.sectors)
            dim = sector.dim
            block = zeros(Float64, length(tau_grid_refl), dim, dim)

            if dim == 2
                # Use omega_matrix for 2x2 block
                for i in 1:2, j in 1:2
                    for (k, tau) in enumerate(tau_grid_refl)
                        block[k, i, j] = exp(-omega_matrix[i,j] * tau)
                    end
                end
            else
                # Use simple exponential for 1x1 blocks
                for (k, tau) in enumerate(tau_grid_refl)
                    block[k, 1, 1] = exp(-0.1234 * tau)
                end
            end

            H_blocks_tau[(sector_idx, sector_idx)] = block
        end

        H_tau = BlockDenseArray{Float64}(basis, flux_zero, H_blocks_tau)

        # Transform to DLR and apply reflection
        H_dlr = tau2dlr_blockdense(dlr_grid_refl, H_tau, tau_grid_refl)
        G_tau = do_reflection_blockdense(dlr_grid_refl, H_dlr)
        G_dlr = tau2dlr_blockdense(dlr_grid_refl, G_tau, tau_grid_refl)

        # Test on fine grid
        tau_grid_test = collect(LinRange(0.0, beta_refl, 10000))
        G_reconstructed = dlr2tau_blockdense(dlr_grid_refl, G_dlr, tau_grid_test)

        # Check N=1 sector (dim=2) - matrix case
        sector_idx_matrix = 2
        G_block_reconstructed = G_reconstructed.blocks[(sector_idx_matrix, sector_idx_matrix)]

        # Compute errors for matrix block
        max_error_matrix = 0.0
        sum_sq_error_matrix = 0.0
        n_elements_matrix = 0
        max_val = 0.0
        max_tau_idx = 1
        max_i, max_j = 1, 1

        for k in 1:length(tau_grid_test)
            tau = tau_grid_test[k]
            for i in 1:2, j in 1:2
                expected = exp(-omega_matrix[i,j] * (beta_refl - tau))
                actual = G_block_reconstructed[k, i, j]
                error = abs(actual - expected)
                max_error_matrix = max(max_error_matrix, error)
                sum_sq_error_matrix += error^2
                n_elements_matrix += 1

                if abs(expected) > abs(max_val)
                    max_val = expected
                    max_tau_idx = k
                    max_i, max_j = i, j
                end
            end
        end

        # Also check dim=1 sectors (scalar case) - N=0 and N=2 sectors
        max_error_scalar = 0.0
        sum_sq_error_scalar = 0.0
        n_elements_scalar = 0
        for sector_idx_scalar in [1, 3]
            G_block_scalar = G_reconstructed.blocks[(sector_idx_scalar, sector_idx_scalar)]
            for k in 1:length(tau_grid_test)
                tau = tau_grid_test[k]
                expected_scalar = exp(-0.1234 * (beta_refl - tau))
                actual_scalar = G_block_scalar[k, 1, 1]
                error = abs(actual_scalar - expected_scalar)
                max_error_scalar = max(max_error_scalar, error)
                sum_sq_error_scalar += error^2
                n_elements_scalar += 1
            end
        end

        # Combined errors
        err_inf = max(max_error_matrix, max_error_scalar)
        err_l2 = sqrt((sum_sq_error_matrix + sum_sq_error_scalar) / (n_elements_matrix + n_elements_scalar))

        actual_at_max = G_block_reconstructed[max_tau_idx, max_i, max_j]

        println("="^40)
        println("Reflection BlockDense Test")
        @printf("Max value of G analytical: %.3e at tau=%.3e with error %.3e\n", max_val, tau_grid_test[max_tau_idx], abs(actual_at_max - max_val))
        @printf("Reflection BlockDense test: ||error||_inf = %.3e, ||error||_2 = %.3e\n", err_inf, err_l2)
        @test err_inf < rtol*100 * exp(maximum(abs.(omega_matrix)) * beta_refl)
        @test err_l2 < rtol*100 * exp(maximum(abs.(omega_matrix)) * beta_refl)
    end

    println("\n" * "="^40)
    println("All BlockDense DLR tests passed!")
    println("="^40)
end