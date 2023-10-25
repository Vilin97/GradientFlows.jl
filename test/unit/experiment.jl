using GradientFlows, Test

@testset "Experiment" begin
    problem = diffusion_problem(2, 10, Blob(blob_eps(2,10)))
    experiment = GradFlowExperiment(problem)
    solve!(experiment)
    compute_errors!(experiment)
    @test experiment.timer isa TimerOutput
    @test experiment.L2_error > 0
    @test experiment.mean_norm_error > 0
    @test experiment.cov_norm_error > 0
    @test experiment.cov_trace_error > 0
end