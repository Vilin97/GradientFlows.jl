using GradientFlows
include("plot.jl")
include("telegram_alerts.jl")

problems = [(anisotropic_landau_problem, 2), (coulomb_landau_mixture_problem, 2), (coulomb_landau_normal_problem, 2), (anisotropic_landau_problem, 3), (coulomb_landau_mixture_problem, 3), (coulomb_landau_normal_problem, 3)]
num_runs = 1
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

### train nn ###
# for (problem, d) in problems
#     train_nn(problem, d, 20000, mlp(d, depth=1); verbose=2)
# end

### generate data ###
# try
#     elapsed = @elapsed run_experiments(problems, ns, num_runs, solvers)
#     sendTelegramMessage("Experiments finished in $elapsed seconds.")
# catch e
#     sendTelegramMessage("Error in experiments.")
#     rethrow(e)
# end
plot_all(problems, ns, solvers)