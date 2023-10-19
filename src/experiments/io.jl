### GradFlowExperiment IO ###
experiment_filename(problem_name, solver, d, n, num_solutions; path=joinpath("data", "experiments")) = joinpath(path, lowercase(problem_name), lowercase(solver), "d_$(d)", "n_$(n)", "runs_$(num_solutions).jld2")
function experiment_filename(experiment::GradFlowExperiment; path="")
    d, n = size(experiment.problem.u0)
    return experiment_filename(experiment.problem.name, "$(experiment.problem.solver)", d, n, experiment.num_solutions; path=path)
end

### Model IO ###
model_filename(problem_name, d, n; path=joinpath("data", "models")) = joinpath(path, lowercase(problem_name), "d_$(d)", "n_$(n).jld2")

save(path, obj) = (mkpath(dirname(path)); JLD2.save_object(path, obj))
load(path) = JLD2.load_object(path)