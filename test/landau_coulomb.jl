using GradientFlows, StableRNGs, Test
using GradientFlows: have_true_dist

n = 2000
d = 2
for solver in [Blob(), SBTM(mlp(d, rng=StableRNG(321), depth=1))]
    @show solver
    problem = coulomb_landau_problem(d, n, solver; rng=StableRNG(123))
    experiment = Experiment(problem)
    result = GradFlowExperimentResult(experiment)

    # @test have_true_dist(experiment) == true
    # This is a very loose bound
    @test result.true_cov_trace_error[1] < 0.9
    @test result.true_cov_norm_error[1] < 0.9
end
