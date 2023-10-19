using GradientFlows

ds = [2,3]
ns = [50,100]
problems = Array{GradFlowProblem,3}(undef, length(ds), length(ns), 3)
for (i,d) in enumerate(ds), (j,n) in enumerate(ns)
    for (k,solver) in enumerate([Exact(), SBTM(mlp(d, depth=1)), Blob(0.16)])
        problem = diffusion_problem(d, n, solver)
        problems[i,j,k] = problem
    end
end
num_solutions = 2
experiment_set = GradFlowExperimentSet(problems, num_solutions)
run_experiment_set!(experiment_set)
@show experiment_set