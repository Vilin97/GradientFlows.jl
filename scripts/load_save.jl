using GradientFlows

problems = [("diffusion", 2), ("diffusion", 5), ("landau", 3), ("landau", 5), ("landau", 10)]
solvers = ["blob", "sbtm", "exact"]
num_runs = 5
ns = 100 * 2 .^ (0:8)
dir = joinpath("data", "dt_0025")

for (problem_name, d) in problems, n in ns, solver_name in solvers, run in 1:num_runs
    @show problem_name, d, n, solver_name, run
    grad_flow_exp = load(experiment_filename(problem_name, d, n, solver_name, run; dir=dir))
    exp = Experiment(grad_flow_exp)
    save(experiment_filename(problem_name, d, n, solver_name, run; dir=dir), exp)
end