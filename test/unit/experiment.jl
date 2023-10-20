using GradientFlows, Test

@testset "Experiment" begin
    problem = diffusion_problem(2, 10, Blob(blob_eps(2,10)))
    experiment = GradFlowExperiment(problem, 1)
    solve!(experiment)
    compute_errors!(experiment)
    @test experiment.num_solutions == length(experiment.solutions)
end