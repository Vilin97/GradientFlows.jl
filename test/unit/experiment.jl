using GradientFlows, Test

@testset "Experiment" begin
    problem = diffusion_problem(2, 10, Blob())
    experiment = GradFlowExperiment(problem, 1)
    solve!(experiment)
    compute_errors!(experiment)
    @test experiment.num_solutions == length(experiment.solutions)
end