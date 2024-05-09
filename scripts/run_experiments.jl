using GradientFlows
include("plot.jl")
include("telegram_alerts.jl")
include("logging.jl")

problems = []
for IC in ["normal", "mixture"], γ in [0, -3], d in [6]
    push!(problems, (landau_problem_factory(d; IC=IC, γ=γ), d))
end

runs = 1:10
ns = 100 * 2 .^ (0:8)
solvers = [SBTM(), Blob()]

### train nn ###
@log @trySendTelegramMessage train_nns(problems, 80000; nn_depth=1, verbose=2)

### generate data ###
@log @trySendTelegramMessage run_experiments(problems, ns, runs, solvers)

### plot ###
@log @trySendTelegramMessage plot_all(problems, ns, solvers)