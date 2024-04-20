using GradientFlows
include("plot.jl")

problems = [(coulomb_landau_normal_problem, 3)]
num_runs = 5
ns = 100 * 2 .^ (0:6)
solvers = [SBTM(), Blob()]

### train nn ###
for (problem, d) in problems
    train_nn(problem, d, 20000, mlp(d, depth=1))
end

### generate data ###
run_experiments(problems, ns, num_runs, solvers)
plot_all(problems, ns, solvers)
