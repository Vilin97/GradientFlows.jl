using GradientFlows
include("plot.jl")
include("telegram_alerts.jl")

problems = [(maxwell_landau_normal_problem, 2), (maxwell_landau_mixture_problem, 2), (coulomb_landau_normal_problem, 2), (coulomb_landau_mixture_problem, 2), (maxwell_landau_normal_problem, 3), (maxwell_landau_mixture_problem, 3), (coulomb_landau_normal_problem, 3), (coulomb_landau_mixture_problem, 3)]
num_runs = 5
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

### train nn ###
try
    elapsed = @elapsed train_nns(problems, 20000; nn_depth=1, verbose=2)
    sendTelegramMessage("NN training finished in $elapsed seconds.")
catch e
    sendTelegramMessage("Error in NN training.")
    rethrow(e)
end

### generate data ###
try
    elapsed = @elapsed run_experiments(problems, ns, num_runs, solvers)
    sendTelegramMessage("Experiments finished in $elapsed seconds.")
catch e
    sendTelegramMessage("Error in experiments.")
    rethrow(e)
end

### plot ###
try
    elapsed = @elapsed plot_all(problems, ns, solvers)
    sendTelegramMessage("Plots finished in $elapsed seconds.")
catch e
    sendTelegramMessage("Error in plots.")
    rethrow(e)
end