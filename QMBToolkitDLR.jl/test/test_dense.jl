using QMBToolkitDLR
using Test
using Printf

@testset "scalar convolution DLR" begin
    beta = 1000.0
    Euv = 1000.0
    rtol = 1e-12
    dlr_grid = DLRGrid(Euv, beta, rtol, true)

    omega_F = 0.1234
    omega_G = -0.5678

    tau_grid = dlr_grid.tau
    F=[kernel_tau(tau, omega_F, beta) for tau in tau_grid]
    G=[kernel_tau(tau, omega_G, beta) for tau in tau_grid]

    F_dlr = tau2dlr(dlr_grid, F, tau_grid)
    G_dlr = tau2dlr(dlr_grid, G, tau_grid)

    # Create workspace and kernel cache
    r = length(dlr_grid.omega)
    workspace = ConvolutionWorkspace(r, 1, 1, length(tau_grid))
    kernel_cache = compute_kernel_cache(dlr_grid)

    H_tau = do_convolution_dlr2tau!(workspace, dlr_grid, kernel_cache, reshape(F_dlr, :, 1, 1), reshape(G_dlr, :, 1, 1))
    H_dlr = tau2dlr(dlr_grid, H_tau, dlr_grid.tau)

    ### calculate error on an equidistant grid
    tau_grid_test = collect(LinRange(0.0, beta, 10000))
    H_reconstructed_test = dlr2tau(dlr_grid, reshape(H_dlr, :), tau_grid_test)
    H_analytical_test = [ (kernel_tau(tau, omega_G, beta)*kernel_tau(0.0, omega_F, beta) - kernel_tau(tau, omega_F, beta)*kernel_tau(0.0, omega_G, beta)) / (omega_F - omega_G) for tau in tau_grid_test]
    err_inf = maximum(abs.(H_reconstructed_test .- H_analytical_test))
    err_l2 = sqrt(sum(abs.(H_reconstructed_test .- H_analytical_test).^2) / length(tau_grid_test))
    
    max_ind = findmax(abs.(H_analytical_test))[2]
    println("="^40)
    println("Scalar Convolution DLR Test")
    @printf("Max value of H analytical: %.3e at tau=%.3e with error %.3e\n", H_analytical_test[max_ind], tau_grid_test[max_ind], abs(H_reconstructed_test[max_ind] - H_analytical_test[max_ind]))
    @printf("Scalar convolution DLR test: ||error||_inf = %.3e, ||error||_2 = %.3e\n", err_inf, err_l2)
    @test err_inf < rtol*100
    @test err_l2 < rtol*100
end

@testset "matrix convolution DLR" begin
    beta = 1000.0
    Euv = 1000.0
    rtol = 1e-12
    dlr_grid = DLRGrid(Euv, beta, rtol, true)

    omega_F = 0.1234
    omega_G = -0.5678

    F_coeffs = [1.0 2.0; 3.0 4.0]
    G_coeffs = [5.0 6.0; 7.0 8.0]

    tau_grid = dlr_grid.tau
    f_tau = [kernel_tau(tau, omega_F, beta) for tau in tau_grid]
    g_tau = [kernel_tau(tau, omega_G, beta) for tau in tau_grid]

    f_dlr = tau2dlr(dlr_grid, f_tau, tau_grid)
    g_dlr = tau2dlr(dlr_grid, g_tau, tau_grid)

    F_dlr = f_dlr .* reshape(F_coeffs, (1, 2, 2))
    G_dlr = g_dlr .* reshape(G_coeffs, (1, 2, 2))

    # Create workspace and kernel cache
    r = length(dlr_grid.omega)
    workspace = ConvolutionWorkspace(r, 2, 2, length(tau_grid))
    kernel_cache = compute_kernel_cache(dlr_grid)

    H_tau = do_convolution_dlr2tau!(workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)
    H_dlr = tau2dlr(dlr_grid, H_tau, dlr_grid.tau)


    tau_grid_test = collect(LinRange(0.0, beta, 10000))
    H_reconstructed_test = dlr2tau(dlr_grid, H_dlr, tau_grid_test)

    h_analytical_scalar = [ (kernel_tau(tau, omega_G, beta)*kernel_tau(0.0, omega_F, beta) - kernel_tau(tau, omega_F, beta)*kernel_tau(0.0, omega_G, beta)) / (omega_F - omega_G) for tau in tau_grid_test]
    
    H_coeffs = F_coeffs * G_coeffs
    H_analytical_test = h_analytical_scalar .* reshape(H_coeffs, (1, 2, 2))

    err_inf = maximum(abs.(H_reconstructed_test .- H_analytical_test))
    err_l2 = sqrt(sum(abs.(H_reconstructed_test .- H_analytical_test).^2) / length(H_analytical_test))
    
    # find the max_ind for the matrix case
    max_ind = findmax(abs.(H_analytical_test))[2]
    tau_indx = max_ind[1]
    # nice printout
    println("="^40)
    println("Matrix Convolution DLR Test")
    @printf("Max value of H analytical: %.3e at tau=%.3e with error %.3e\n", H_analytical_test[tau_indx, max_ind[2], max_ind[3]], tau_grid_test[tau_indx], abs(H_reconstructed_test[tau_indx, max_ind[2], max_ind[3]] - H_analytical_test[tau_indx, max_ind[2], max_ind[3]]))
    @printf("Matrix Convolution DLR test: ||error||_inf = %.3e, ||error||_2 = %.3e\n", err_inf, err_l2)
    @test err_inf < rtol*100 * maximum(abs.(F_coeffs)) * maximum(abs.(G_coeffs))
    @test err_l2 < rtol*100 * maximum(abs.(F_coeffs)) * maximum(abs.(G_coeffs))
end

@testset "reflection DLR" begin
    beta = 10.0
    Euv = 10.0
    rtol = 1e-12
    dlr_grid = DLRGrid(Euv, beta, rtol, true)

    omega_matrix = [0.1234 0.5678; -0.9012 -0.3456] # All positive omegas

    tau_grid = dlr_grid.tau
    r = length(dlr_grid.omega)

    H_dlr = zeros(Float64, r, 2, 2)
    for i in 1:2
        for j in 1:2
            h_tau = [exp(-omega_matrix[i,j] * tau) for tau in tau_grid]
            H_dlr[:, i, j] = tau2dlr(dlr_grid, h_tau, tau_grid)
        end
    end
    
    H_tau = dlr2tau(dlr_grid, H_dlr, dlr_grid.tau)
    G_tau = do_reflection_tau2tau(dlr_grid, H_tau)
    G_dlr = tau2dlr(dlr_grid, G_tau, dlr_grid.tau)

    tau_grid_test = collect(LinRange(0.0, beta, 10000))
    G_reconstructed_test = dlr2tau(dlr_grid, G_dlr, tau_grid_test)
    
    G_analytical_test = zeros(Float64, length(tau_grid_test), 2, 2)
    for i in 1:2
        for j in 1:2
            G_analytical_test[:, i, j] = [exp(-omega_matrix[i,j] * (beta - tau)) for tau in tau_grid_test]
        end
    end

    err_inf = maximum(abs.(G_reconstructed_test .- G_analytical_test))
    err_l2 = sqrt(sum(abs.(G_reconstructed_test .- G_analytical_test).^2) / length(G_analytical_test))

    max_ind = findmax(abs.(G_analytical_test))[2]
    tau_indx = max_ind[1]
    println("="^40)
    println("Reflection DLR Test")
    @printf("Max value of G analytical: %.3e at tau=%.3e with error %.3e\n", G_analytical_test[tau_indx, max_ind[2], max_ind[3]], tau_grid_test[tau_indx], abs(G_reconstructed_test[tau_indx, max_ind[2], max_ind[3]] - G_analytical_test[tau_indx, max_ind[2], max_ind[3]]))
    @printf("Reflection DLR test: ||error||_inf = %.3e, ||error||_2 = %.3e\n", err_inf, err_l2)
    @test err_inf < rtol*100 * maximum(abs.(G_analytical_test))
    @test err_l2 < rtol*100 * maximum(abs.(G_analytical_test))
end

@testset "kernel_tau" begin
    beta = 10.0
    tau = 2.0
    
    # Test case 1: omega > 0
    omega1 = 1.5
    expected1 = exp(-tau * omega1) / (1 + exp(-beta * omega1))
    @test kernel_tau(tau, omega1, beta) ≈ expected1
    
    # Test case 2: omega < 0
    omega2 = -1.5
    expected2 = exp(-(tau - beta) * omega2) / (1 + exp(beta * omega2))
    @test kernel_tau(tau, omega2, beta) ≈ expected2
end