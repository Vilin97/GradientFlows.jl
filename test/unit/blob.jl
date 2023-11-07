using GradientFlows, Distributions, LinearAlgebra, StableRNGs, Test
using GradientFlows: score, initialize, update!

rng = StableRNG(123)
struct DummyIntegrator
    u::Matrix{Float64}
end

@testset "Blob tests" begin
    d, n = 2, 10
    dist = MvNormal(1.0 * I(d))
    u = rand(rng, dist, n)
    score_values = score(dist, u)

    # test initialize
    solver = initialize(Blob(blob_epsilon(d, n)), u, score_values)
    @test solver.score_values == score_values

    # test that update! is idempotent
    integrator = DummyIntegrator(u)
    update!(solver, integrator)
    score_values = copy(solver.score_values)
    update!(solver, integrator)
    @test solver.score_values == score_values

    # Now we change u
    old_u = copy(u)
    u .-= 0.01 * solver.score_values
    update!(solver, integrator)
    @test solver.score_values != score_values
end