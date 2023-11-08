using GradientFlows, Test, TimerOutputs, StableRNGs

@testset "Experiment" begin
    problem = diffusion_problem(2, 100, Blob(blob_epsilon(2, 10)); rng=StableRNG(1))
    experiment = GradFlowExperiment(problem)
    solve!(experiment)
    result = GradFlowExperimentResult(experiment)
    @test result.L2_error == Lp_error(experiment.solution[end], true_dist(problem, problem.tspan[2]); p=2)
    @test "$experiment" isa String # just to get coverage on the show method

    problem2 = diffusion_problem(2, 100, Exact(); rng=StableRNG(1))
    experiment2 = GradFlowExperiment(problem2)
    solve!(experiment2)
    result2 = GradFlowExperimentResult(experiment2)
    @test !(result.L2_error ≈ result2.L2_error)
end