using GradientFlows, StableRNGs, Test
using GradientFlows: have_true_dist

n = 2000
d = 2
for problem_f in [coulomb_landau_normal_problem, coulomb_landau_mixture_problem]
    for solver in [Blob(), SBTM(mlp(d, rng=StableRNG(321), depth=1))]
        @show solver
        problem = problem_f(d, n, solver; rng=StableRNG(123))
        experiment = Experiment(problem)
        result = GradFlowExperimentResult(experiment)

        # This is a very loose bound
        @test result.true_cov_trace_error[1] < 0.1
        @test result.true_cov_norm_error[1] < 0.1
    end
end