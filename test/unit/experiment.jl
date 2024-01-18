using GradientFlows, Test, TimerOutputs, StableRNGs

@testset "Experiment" begin
    problem = diffusion_problem(2, 100, Blob(); rng=StableRNG(1))
    experiment = Experiment(problem)
    result = GradFlowExperimentResult(experiment)
    @test result.L2_error[1] == Lp_error(experiment.solution[end], true_dist(problem, problem.tspan[2]); p=2)
    @test "$experiment" isa String # just to get coverage on the show method

    problem2 = diffusion_problem(2, 100, Exact(); rng=StableRNG(1))
    experiment2 = Experiment(problem2)
    result2 = GradFlowExperimentResult(experiment2)
    @test !(result.L2_error[1] ≈ result2.L2_error[1])

    # for coverage
    @test "$experiment" isa String
end