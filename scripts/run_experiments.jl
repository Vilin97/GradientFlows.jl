using GradientFlows
include("plot.jl")

problems = [(coulomb_landau_normal_problem, 2), (coulomb_landau_mixture_problem, 2), (anisotropic_landau_problem, 2)]
num_runs = 1
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

### train nn ###
# for problem in problems
#     train_nn(problem..., 20000, mlp(2, depth=1))
# end

### generate data ###
run_experiments(problems, ns, num_runs, solvers)
plot_all(problems, ns, solvers)
