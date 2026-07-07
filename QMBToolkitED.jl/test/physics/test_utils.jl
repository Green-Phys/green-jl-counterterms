module PhysicsTestUtils

using Test
using QMBToolkitED
using LinearAlgebra

export run_physics_test, physics_models

include("models.jl")

"""
    run_physics_test(model)

Run all available physics tests for a model with QN_Trivial, QN_N, and QN_NSz.
Builds and solves the Hamiltonian once per QN type, then runs verify_Z and verify_greens.
"""
function run_physics_test(model)
    nflavors = length(model.flavor_qns_n)
    qn_configs = [
        ("QN_Trivial", [QN_Trivial() for _ in 1:nflavors]),
        ("QN_N", model.flavor_qns_n),
        ("QN_NSz", model.flavor_qns_sz),
    ]

    for (label, flavor_qns) in qn_configs
        @testset "$label" begin
            basis = Basis(flavor_qns)
            H, annh = build_hamiltonian(basis, IOBuffer(model.hopping_str), IOBuffer(model.coulomb_str))
            result = solve_hamiltonian(H)

            if model.verify_Z !== nothing
                evals = to_vector(result.eigenvalues)
                model.verify_Z(evals)
            end

            if model.verify_greens !== nothing
                model.verify_greens(result, annh)
            end
        end
    end
end

end # module
