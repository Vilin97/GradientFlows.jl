using GradientFlows, StableRNGs, Test
using GradientFlows: have_true_dist

n = 2000
d = 2
for solver in [Blob(), SBTM(mlp(d, rng=StableRNG(321), depth=2))]
    @show solver
    problem = coulomb_landau_problem(d, n, solver; rng=StableRNG(123))
    experiment = Experiment(problem)
    result = GradFlowExperimentResult(experiment)

    @test have_true_dist(experiment) == false
    # TODO
    # @test result.true_cov_trace_error[1] < 0.5
    # @test result.true_cov_norm_error[1] < 0.5
end
