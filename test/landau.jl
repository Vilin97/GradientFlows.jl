using GradientFlows, StableRNGs, Test
using GradientFlows: have_true_dist

# isotropic initial condition
n = 2000
for d in [3, 5]
    for solver in [Exact(), SBTM(mlp(d, rng=StableRNG(321), depth=2)), Blob()]
        problem = landau_problem(d, n, solver; rng=StableRNG(123))
        result = GradFlowExperimentResult(Experiment(problem))

        @test result.L2_error < 0.05
        @test result.true_mean_error < 0.05
        @test result.true_cov_trace_error < 0.5
        @test result.true_cov_norm_error < 0.5
    end
end

# anisotropic initial condition
n = 2000
d = 3
for solver in [SBTM(mlp(d, rng=StableRNG(321), depth=2)), Blob()]
    problem = landau_problem(d, n, solver; rng=StableRNG(123), isotropic=false)
    experiment = Experiment(problem)
    empirical_covariance_start = emp_cov(experiment.solution[1])
    empirical_covariance_end = emp_cov(experiment.solution[end])
    @test have_true_dist(experiment) == false
    @test problem.covariance(problem.tspan[1], problem.params) == cov(problem.ρ0)
    @test problem.covariance(problem.tspan[2], problem.params) ≈ empirical_covariance_start atol=0.1
    @test problem.covariance(problem.tspan[2], problem.params) ≈ empirical_covariance_end atol=0.1

end