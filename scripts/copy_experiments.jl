using GradientFlows, Accessors, Distributions, LinearAlgebra
run = 1
d = 2
problem_name = "coulomb_landau"
dir = "data"
for n in 100 * 2 .^ (0:7), solver_name in ["blob", "sbtm"]
    @show n, solver_name
    experiment = load(experiment_filename(problem_name, d, n, solver_name, run; dir=dir))
    
    problem = coulomb_landau_problem(d, 100, Blob())
    true_dist_ = [problem.ρ(t, nothing) for t in experiment.saveat]
    new_experiment = @set experiment.true_dist = true_dist_
    save(experiment_filename(problem_name, d, n, solver_name, run; dir=dir), new_experiment)
end