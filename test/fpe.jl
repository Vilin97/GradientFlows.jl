using GradientFlows, StableRNGs, Test

d = 2
n = 2000
for solver in [Exact(), SBTM(mlp(d, depth=1, rng=StableRNG(321))), Blob()]
    problem = fpe_problem(d, n, solver; rng=StableRNG(123))
    result = GradFlowExperimentResult(Experiment(problem))
    
    @test result.L2_error < 0.05
	@test result.true_mean_error < 0.05
	@test result.true_cov_trace_error < 0.5
	@test result.true_cov_norm_error < 0.5
end
