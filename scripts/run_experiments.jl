using GradientFlows
include("plot.jl")
include("telegram_alerts.jl")

problems = []
for IC in ["normal", "mixture"], γ in [0, -3], d in [2, 3]
    push!(problems, (landau_problem_factory(d; IC=IC, γ=γ), d))
end

num_runs = 5
ns = 100 * 2 .^ (7:8)
solvers = [SBTM(), Blob()]

### train nn ###
@trySendTelegramMessage train_nns(problems, 80000; nn_depth=1, verbose=2)

### generate data ###
@trySendTelegramMessage run_experiments(problems, ns, num_runs, solvers)

### plot ###
@trySendTelegramMessage plot_all(problems, ns, solvers)