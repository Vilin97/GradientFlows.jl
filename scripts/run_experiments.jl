using GradientFlows
include("plot.jl")
include("telegram_alerts.jl")
include("logging.jl")

problems = []
for IC in ["normal", "mixture"], γ in [0, -3], d in [2,3,6]
    push!(problems, (landau_problem_factory(d; IC=IC, γ=γ), d))
end
for d in [2,3,6]
    push!(problems, (landau_problem, d))
end

runs = 1:1
ns = 100 * 2 .^ (0:6)
solvers = [ASBTM(verbose=2)]

### train nn ###
# @log @trySendTelegramMessage train_nns([(landau_problem, d) for d in [2,3,6]], 80000; nn_depth=1, verbose=2)

### generate data ###
@log @trySendTelegramMessage run_experiments(problems, ns, runs, solvers)

### plot ###
# @log @trySendTelegramMessage plot_all(problems, ns, solvers)