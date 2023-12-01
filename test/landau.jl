using GradientFlows, StableRNGs, Test

n = 2000
for d in [3, 5]
    for solver in [Exact(), SBTM(mlp(d, rng=StableRNG(321), depth=2); logger=Logger(0)), Blob()]
        problem = landau_problem(d, n, solver; rng=StableRNG(123))
        result = GradFlowExperimentResult(Experiment(problem))
    
        @test result.L2_error < 0.05
        @test result.true_mean_error < 0.05
        @test result.true_cov_trace_error < 0.5
        @test result.true_cov_norm_error < 0.5
    end
end