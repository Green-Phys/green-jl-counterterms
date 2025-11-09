using QMBToolkitSOEXCA
using Test
using Lehmann
using Printf

include("../src/util/convolution_dlr.jl")

@testset "scalar convolution DLR" begin
    beta = 1000.0
    Euv = 1000.0
    rtol = 1e-12
    dlr_grid = DLRGrid(Euv, beta, rtol, true)

    omega_F = 0.1234
    omega_G = -0.5678

    tau_grid = dlr_grid.tau
    F=[kernel_dlr(tau, omega_F, beta) for tau in tau_grid]
    G=[kernel_dlr(tau, omega_G, beta) for tau in tau_grid]

    F_dlr = tau2dlr(dlr_grid, F, tau_grid)
    G_dlr = tau2dlr(dlr_grid, G, tau_grid)

    H_dlr = do_convolution_dlr(dlr_grid, reshape(F_dlr, :, 1, 1), reshape(G_dlr, :, 1, 1))
    
    ### calculat error on an equidistant grid
    tau_grid_test = collect(LinRange(0.0, beta, 10000))
    H_reconstructed_test = dlr2tau(dlr_grid, reshape(H_dlr, :,), tau_grid_test)
    H_analytical_test = [ (kernel_dlr(tau, omega_G, beta)*kernel_dlr(0.0, omega_F, beta) - kernel_dlr(tau, omega_F, beta)*kernel_dlr(0.0, omega_G, beta)) / (omega_F - omega_G) for tau in tau_grid_test]
    err_inf = maximum(abs.(H_reconstructed_test .- H_analytical_test))
    err_l2 = sqrt(sum(abs.(H_reconstructed_test .- H_analytical_test).^2) / length(tau_grid_test))
    
    max_ind = findmax(abs.(H_analytical_test))[2]
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
    f_tau = [kernel_dlr(tau, omega_F, beta) for tau in tau_grid]
    g_tau = [kernel_dlr(tau, omega_G, beta) for tau in tau_grid]

    f_dlr = tau2dlr(dlr_grid, f_tau, tau_grid)
    g_dlr = tau2dlr(dlr_grid, g_tau, tau_grid)

    F_dlr = f_dlr .* reshape(F_coeffs, (1, 2, 2))
    G_dlr = g_dlr .* reshape(G_coeffs, (1, 2, 2))

    H_dlr = do_convolution_dlr(dlr_grid, F_dlr, G_dlr)

    tau_grid_test = collect(LinRange(0.0, beta, 10000))
    H_reconstructed_test = dlr2tau(dlr_grid, H_dlr, tau_grid_test)

    h_analytical_scalar = [ (kernel_dlr(tau, omega_G, beta)*kernel_dlr(0.0, omega_F, beta) - kernel_dlr(tau, omega_F, beta)*kernel_dlr(0.0, omega_G, beta)) / (omega_F - omega_G) for tau in tau_grid_test]
    
    H_coeffs = F_coeffs * G_coeffs
    H_analytical_test = h_analytical_scalar .* reshape(H_coeffs, (1, 2, 2))

    err_inf = maximum(abs.(H_reconstructed_test .- H_analytical_test))
    err_l2 = sqrt(sum(abs.(H_reconstructed_test .- H_analytical_test).^2) / length(H_analytical_test))
    
    # find the max_ind for the matrix case
    max_ind = findmax(abs.(H_analytical_test))[2]
    tau_indx = max_ind[1]
    @printf("Max value of H analytical: %.3e at tau=%.3e with error %.3e\n", H_analytical_test[tau_indx, max_ind[2], max_ind[3]], tau_grid_test[tau_indx], abs(H_reconstructed_test[tau_indx, max_ind[2], max_ind[3]] - H_analytical_test[tau_indx, max_ind[2], max_ind[3]]))
    @printf("Matrix Convolution DLR test: ||error||_inf = %.3e, ||error||_2 = %.3e\n", err_inf, err_l2)
    @test err_inf < rtol*100 * maximum(abs.(F_coeffs)) * maximum(abs.(G_coeffs))
    @test err_l2 < rtol*100 * maximum(abs.(F_coeffs)) * maximum(abs.(G_coeffs))
end
