using GradientFlows

### generate data ###
problems = ALL_PROBLEMS
num_runs = 5
ns = 100 * 2 .^ (0:8)
solvers = [Blob(), NPF()]

# run_experiments(problems, ns, num_runs, [NPF(), Blob()])

include("plot.jl")
@time plot_all(problems, ns, solvers; dir="data", scale=:identity);
