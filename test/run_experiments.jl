using GradientFlows, Test

problems = [(diffusion_problem, 2), (landau_problem, 3), (fpe_problem, 2)]
models = [mlp(2; depth=1), mlp(3; depth=2), mlp(2; depth=1)]
num_runs = 1
ns = 10 * 2 .^ (0:6)
solvers = ALL_SOLVERS
dir = "data_test"

# train models
for ((problem, d), model) in zip(problems, models)
    train_nn(problem, d, ns[end], model; init_max_iterations=10^4, dir=dir, verbose=1)
end

# generate data
run_experiments(problems, ns, num_runs, solvers; verbose=1, dir=dir)

# plot
include("../scripts/plot.jl")
plot_all(problems, ns, solvers; dir=dir)

# test that errors are small
metrics = [:L2_error, :true_mean_error, :true_cov_trace_error, :true_cov_norm_error]

for (problem_name, d) in [("diffusion", 2), ("landau", 3), ("fpe", 2)], metric in metrics
    metric_matrix = load_metric(problem_name, d, ns, name.(solvers), metric; dir=dir)
    @test maximum(metric_matrix[end, :]) < 0.5
end

rm(dir, recursive=true)