using GradientFlows, Distributions, LinearAlgebra, StableRNGs, Test
using GradientFlows: score, initialize, update!

rng = StableRNG(123)

@testset "Blob tests" begin
    d, n = 2, 10
    dist = MvNormal(1.0 * I(d))
    u = rand(rng, dist, n)
    score_values = score(dist, u)

    # test initialize
    solver = initialize(Blob(), u, score_values, "dummy_problem_name")
    @test solver.score_values == score_values

    # test that update! is idempotent
    prob = (solver=solver, f! = (args...) -> nothing, )
    integrator = (u=u, p=prob, t=0., )
    update!(solver, integrator)
    score_values = copy(solver.score_values)
    update!(solver, integrator)
    @test solver.score_values == score_values

    # Now we change u
    old_u = copy(u)
    u .-= 0.01 * solver.score_values
    update!(solver, integrator)
    @test solver.score_values != score_values

    # for coverage
    @test "$(Blob())" == "Blob"
end