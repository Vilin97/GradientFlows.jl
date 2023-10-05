using GradientFlows, Test, StableRNGs, Distributions, CUDA

rng = StableRNG(123)

function test_prob(problem)
    solution = solve(problem)
    u = solution[end]
    end_dist = true_dist(problem, problem.tspan[2])
    @test emp_mean(u) ≈ mean(end_dist) atol = 0.1
    @test emp_cov(u) ≈ cov(end_dist) rtol = 0.1
    @test Lp_error(u, end_dist; p=2) ≈ 0 atol = 0.1
end

d = 2
n = 5000
problem = diffusion_problem(d, n, Exact(); rng=rng)
test_prob(problem)
CUDA.functional() && test_prob(cu(problem))