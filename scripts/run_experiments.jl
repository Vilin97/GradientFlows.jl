using GradientFlows

### generate data ###
problems = [(anisotropic_landau_problem, 3), (anisotropic_landau_problem, 5), (anisotropic_landau_problem, 10)]
num_runs = 5
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

run_experiments(problems, ns, num_runs, solvers)

include("plot.jl")
@time plot_all(ALL_PROBLEMS, ns, [SBTM(), Blob()]; dir="data");
