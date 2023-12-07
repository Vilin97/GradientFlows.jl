using GradientFlows, TimerOutputs, StableRNGs

### generate data ###
problems = ALL_PROBLEMS
num_runs = 5
ns = 100 * 2 .^ (0:8)
solvers = [Blob()]

run_experiments(problems, ns, num_runs, solvers; dir="data/eps_reduced_x10")
