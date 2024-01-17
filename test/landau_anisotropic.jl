using GradientFlows, StableRNGs, Test
using GradientFlows: have_true_dist

n = 2000
d = 3
for solver in [SBTM(mlp(d, rng=StableRNG(321), depth=2)), Blob()]
    @show solver
    problem = anisotropic_landau_problem(d, n, solver; rng=StableRNG(123))
    experiment = Experiment(problem)
    @test have_true_dist(experiment) == false
    @test problem.covariance(problem.tspan[1], problem.params) ≈ cov(problem.ρ0)
    @test problem.covariance(problem.tspan[1], problem.params) ≈ emp_cov(experiment.solution[1]) atol=0.1
    @test problem.covariance(problem.tspan[2], problem.params) ≈ emp_cov(experiment.solution[end]) atol=0.15
end
