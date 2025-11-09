using QMBToolkitSOEXCA
using Test
using Lehmann
using Printf

include("../src/util/convolution_dlr.jl")

@testset "convolution_dlr.jl" begin
    beta = 10.0
    Euv = 10.0
    dlr_grid = DLRGrid(Euv, beta, 1e-10, true)

    omega0 = 4
    omega1 = 2

    # F(tau) = exp(-omega0 * tau)
    F_tau = exp.(-omega0 .* dlr_grid.tau)
    F_dlr = tau2dlr(dlr_grid, F_tau, dlr_grid.tau)
    F_dlr = reshape(F_dlr, (dlr_grid.size, 1, 1))

    # G(tau) = exp(-omega1 * tau)
    G_tau = exp.(-omega1 .* dlr_grid.tau)
    G_dlr = tau2dlr(dlr_grid, G_tau, dlr_grid.tau)
    G_dlr = reshape(G_dlr, (dlr_grid.size, 1, 1))

    H_dlr = do_convolution_dlr(dlr_grid, F_dlr, G_dlr)

    # H(tau) = (exp(-omega1 * tau) - exp(-omega0 * tau)) / (omega0 - omega1)
    H_tau = (exp.(-omega1 .* dlr_grid.tau) .- exp.(-omega0 .* dlr_grid.tau)) ./ (omega0 - omega1)
    H_dlr_expected = tau2dlr(dlr_grid, H_tau, dlr_grid.tau)
    # find the index of the maximum error
    index_max_error = argmax(abs.(H_dlr[:, 1, 1] .- H_dlr_expected))
    @printf("Maximum error at index %d: \n", index_max_error)
    @printf("H_dlr: %e, H_dlr_expected: %e, error: %e\n", H_dlr[index_max_error, 1, 1], H_dlr_expected[index_max_error], abs(H_dlr[index_max_error, 1, 1] - H_dlr_expected[index_max_error]))
    @test H_dlr[:, 1, 1] ≈ H_dlr_expected atol=1e-3
end
