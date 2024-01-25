using GradientFlows

### generate data ###
problems = ALL_PROBLEMS
num_runs = 5
ns = 100 * 2 .^ (0:4)
solvers = [Blob(), SBTM()]

run_experiments(problems, ns, num_runs, [SBTM(), Blob()])

# include("plot.jl")
# @time plot_all(problems, ns, solvers; dir="data", scale=:identity);
