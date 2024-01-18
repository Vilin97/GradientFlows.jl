using GradientFlows, StableRNGs, Test
using GradientFlows: have_true_dist

n = 2000
d = 3
for solver in [SBTM(mlp(d, rng=StableRNG(321), depth=2)), Blob()]
    @show solver
    problem = anisotropic_landau_problem(d, n, solver; rng=StableRNG(123))
    experiment = Experiment(problem)
    result = GradFlowExperimentResult(Experiment(problem))
    
    @test have_true_dist(experiment) == false
    @test result.L2_error[1] < 0.05
    @test result.true_mean_error[1] < 0.05
    @test result.true_cov_trace_error[1] < 0.5
    @test result.true_cov_norm_error[1] < 0.5
end
