using GradientFlows, StableRNGs, Test, Distributions

@testset "GradFlowProblem tests" begin
    problem = diffusion_problem(2, 10, Exact(); rng=StableRNG(123))
    ρ0 = problem.ρ0
    @test "$problem" isa String
    @test true_dist(problem, problem.tspan[1]) isa MvNormal
    @test true_dist(problem, problem.tspan[1]) == ρ0
    @test set_solver(problem, Blob(blob_eps(2,10))) == diffusion_problem(2, 10, Blob(blob_eps(2,10)); rng=StableRNG(123))

    old_u0 = copy(problem.u0)
    resample!(problem; rng=StableRNG(1234))
    @test problem.u0 != old_u0
    @test problem.solver.score_values == score(ρ0, problem.u0)
end