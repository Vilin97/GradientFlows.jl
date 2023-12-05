using GradientFlows, TimerOutputs, StableRNGs

### generate data ###
problems = [(fpe_problem, 2), (fpe_problem, 5), (fpe_problem, 10)]
num_runs = 5
ns = 100 * 2 .^ (0:8)
solvers = ALL_SOLVERS

# run_experiments(problems, ns, num_runs, solvers)

include("plot.jl")
@time plot_all(ALL_PROBLEMS, ns, ALL_SOLVERS; dir="data/dt_0025");
