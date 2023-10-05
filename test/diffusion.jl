using GradientFlows, Test, StableRNGs, Distributions

rng = StableRNG(123)

d = 2
n = 10000
problem = diffusion_problem(d, n, Exact(); rng=rng)
@test problem isa GradFlowProblem{D,Float32} where {D}

solution = solve(problem)
u = solution[end]
@test size(u) == (d, n)
end_dist = true_dist(problem, problem.tspan[2])
@test emp_mean(u) ≈ mean(end_dist) atol = 0.1
@test emp_cov(u) ≈ cov(end_dist) rtol = 0.1
@test Lp_error(u, end_dist; p=2) ≈ 0 atol = 0.1