using GradientFlows, Test, StableRNGs, Distributions, Random

function test_prob(problem; debug=false, time=false)
    if time
        @time solution = solve(problem; saveat=problem.tspan[2])
    else
        solution = solve(problem; saveat=problem.tspan[2])
    end
    u = solution[end]
    end_dist = true_dist(problem, problem.tspan[2])
    if debug
        @show problem
        @show emp_mean(u)
        @show emp_cov(u), cov(end_dist)
        @show Lp_error(u, end_dist; p=2)
    end
    @test emp_mean(u) ≈ mean(end_dist) atol = 0.05
    @test emp_cov(u) ≈ cov(end_dist) atol = 1.
    @test Lp_error(u, end_dist; p=2) ≈ 0 atol = 0.05
end

d = 2
n = 2000
for solver in [Exact(), SBTM(mlp(d); logger=Logger(1)), Blob(0.16)]
    problem = diffusion_problem(d, n, solver; rng=StableRNG(123))
    test_prob(problem)
end