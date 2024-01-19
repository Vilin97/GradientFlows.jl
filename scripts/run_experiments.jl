using GradientFlows

### generate data ###
problems = [(anisotropic_landau_problem, 3), (anisotropic_landau_problem, 5), (anisotropic_landau_problem, 10)]
num_runs = 5
ns = 100 * 2 .^ (0:8)
solvers = [Blob(), SBTM()]

run_experiments(problems, 100 * 2 .^ (7:8), num_runs, solvers)

include("plot.jl")
@time plot_all(problems, ns, solvers; dir="data", scale=:identity);
