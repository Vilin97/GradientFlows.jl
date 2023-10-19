using GradientFlows, Flux, Test
using GradientFlows: model_filename, experiment_filename, GradFlowExperiment, solve!, compute_errors!
include("../testutils.jl")

@testset "chain IO" begin
    d = 2
    n = 10
    s = Chain(Dense(d=>d))
    path = model_filename("test_problem", d, n)
    save(path, s)
    s_loaded = load(path)
    x = rand(Float32, d)
    @test s(x) == s_loaded(x)

    path_prefix = splitpath(path)[1]
    rm(path_prefix, recursive=true)
end
@testset "experiment IO" begin
    problem = diffusion_problem(2, 10, Blob())
    experiment = GradFlowExperiment(problem, 1)
    path = experiment_filename(experiment)
    save(path, experiment)
    experiment_loaded = load(path)
    @test experiment_loaded.problem == experiment.problem
    path_prefix = splitpath(path)[1]
    rm(path_prefix, recursive=true)
end