using GradientFlows

### generate data ###
problems = [(diffusion_problem, 2), (diffusion_problem, 5), (diffusion_problem, 10)]
num_runs = 5
ns = 100 * 2 .^ (0:8)
solvers = ALL_SOLVERS

# run_experiments(problems, ns, num_runs, solvers)

include("plot.jl")
@time plot_all(ALL_PROBLEMS, ns, [Blob()]; dir="data/eps_reduced_x10");
