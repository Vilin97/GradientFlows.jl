using GradientFlows

d = 5
n = 4*10^4
@time problem = landau_problem(d, n, SBTM(mlp(5, depth=2), logger=Logger(2), init_max_iterations=10^6))
save(model_filename(problem.name, d, n), problem.solver.s)