using GradientFlows

### train nn ###
train_nn(coulomb_landau_problem, 2, 20000, mlp(2, depth=2); verbose=1, init_max_iterations=10^5, dir="data")

### generate data ###
problems = [(coulomb_landau_problem, 2)]
num_runs = 1
ns = 100 * 2 .^ (0:7)
solvers = [Blob(), SBTM()]
run_experiments(problems, ns, num_runs, solvers)

# include("plot.jl")
# @time plot_all(problems, ns, solvers; dir="data");
