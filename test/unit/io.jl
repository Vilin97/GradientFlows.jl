using GradientFlows, Flux, Test, TimerOutputs
include("../testutils.jl")

dir = "data_test"
@testset "IO" begin
@testset "chain" begin
    d = 2
    n = 10
    s = Chain(Dense(d => d))
    path = model_filename("test_problem", d, n; dir=dir)
    save(path, s)
    s_loaded = load(path)
    x = rand(Float32, d)
    @test s(x) == s_loaded(x)

    other_s = Chain(Dense(d => d))
    path2 = model_filename("test_problem", d, n + 1; dir=dir)
    save(path2, other_s)
    s_loaded = best_model("test_problem", d)
    @test s(x) != s_loaded(x)
    @test other_s(x) == s_loaded(x)

    path_prefix = splitpath(path)[1]
    rm(path_prefix, recursive=true)
end
@testset "experiment" begin
    problem = diffusion_problem(2, 10, Blob(blob_epsilon(2, 10)))
    experiment = Experiment(problem)
    path = experiment_filename(experiment, 1; dir=dir)
    save(path, experiment)
    experiment_loaded = load(path)
    @test experiment_loaded.problem == experiment.problem
    path_prefix = splitpath(path)[1]
    rm(path_prefix, recursive=true)
end

@testset "experiment result" begin
    problem = diffusion_problem(2, 10, Blob(blob_epsilon(2, 10)))
    experiment = Experiment(problem)
    result = GradFlowExperimentResult(experiment)
    path = experiment_result_filename(experiment, 1; dir=dir)
    save(path, result)
    result_loaded = load(path)
    @test result_loaded == result
    for metric in fieldnames(GradFlowExperimentResult)
        @test load_metric("diffusion", 2, [10], ["Blob"], metric)[1] == getfield(result, metric)
    end
    path_prefix = splitpath(path)[1]
    rm(path_prefix, recursive=true)
end

@testset "timer" begin
    timer = TimerOutput()
    @timeit timer "test" sleep(0.5)
    path = timer_filename(dir=dir)
    save(path, timer)
    timer_loaded = load(path)
    @test timer_loaded isa TimerOutput
    path_prefix = splitpath(path)[1]
    rm(path_prefix, recursive=true)
end
end