using GradientFlows
include("plot.jl")
include("telegram_alerts.jl")

problems = []
for IC in ["normal", "mixture"], γ in [0, -3], covariance_scale in [1, 100], d in [2, 3]
    push!(problems, (landau_problem_factory(d; IC=IC, γ=γ, covariance_scale=covariance_scale), d))
end

num_runs = 5
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

### train nn ###
@trySendTelegramMessage train_nns(problems, 40000; nn_depth=1, verbose=2)

### generate data ###
@trySendTelegramMessage run_experiments(problems, ns, num_runs, solvers)

### plot ###
@trySendTelegramMessage plot_all(problems, ns, solvers)