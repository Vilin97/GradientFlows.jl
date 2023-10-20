using GradientFlows

d = 2
ns = [50,100]
num_solutions = 2
experiment_set = GradFlowExperimentSet(diffusion_problem, d, ns, num_solutions; model=mlp(d, depth=1))
run_experiment_set!(experiment_set)
print(experiment_set)