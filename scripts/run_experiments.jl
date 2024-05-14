using GradientFlows
include("plot.jl")
include("telegram_alerts.jl")
include("logging.jl")

problems = []
ds = [3,6]
for IC in ["normal"], γ in [0, -3], d in ds
    push!(problems, (landau_problem_factory(d; IC=IC, γ=γ), d))
end
for d in ds
    push!(problems, (landau_problem, d))
end

runs = 1:10
ns = 100 * 2 .^ (0:7)
solvers = [SBTM()]

### train nn ###
@log @trySendTelegramMessage train_nns([(landau_problem, d) for d in ds], 80000; nn_depth=2, verbose=2)

### generate data ###
@log @trySendTelegramMessage run_experiments(problems, ns, runs, solvers)

### plot ###
@log @trySendTelegramMessage plot_all(problems, ns, [SBTM(), Blob()])