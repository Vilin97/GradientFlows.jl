using GradientFlows, Test, TimerOutputs

@testset "Experiment" begin
    problem = diffusion_problem(2, 10, Blob(blob_eps(2,10)))
    experiment = GradFlowExperiment(problem)
    solve!(experiment)
    compute_errors!(experiment)
    @test experiment.timer isa TimerOutput
    @test experiment.L2_error == Lp_error(experiment.solution[end], true_dist(problem, problem.tspan[2]); p=2)
end