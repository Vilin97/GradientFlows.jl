using GradientFlows, StableRNGs, Test, Distributions

@testset "GradFlowProblem tests" begin
    problem = diffusion_problem(2, 10, Exact(); rng=StableRNG(1))
    ρ0 = problem.ρ0
    @test "$problem" isa String
    @test true_dist(problem, problem.tspan[1]) isa MvNormal
    @test true_dist(problem, problem.tspan[1]) == ρ0

    old_u0 = copy(problem.u0)
    new_u0 = rand(StableRNG(2), ρ0, 10)
    set_u0!(problem, new_u0)
    @test problem.u0 == new_u0
    @test problem.u0 != old_u0
    @test problem.solver.score_values == score(ρ0, problem.u0)
end