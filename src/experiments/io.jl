### generic ###
save(path, obj) = (mkpath(dirname(path)); JLD2.save_object(path, obj))
load(path) = JLD2.load_object(path)

### experiment ###
experiment_filename(problem_name, d, n, solver, id; dir="data") = joinpath(dir, "experiments", lowercase(problem_name), "d_$d", "n_$n", lowercase(solver), "$(id).jld2")
function experiment_filename(experiment::GradFlowExperiment, id; kwargs...)
    d, n = size(experiment.problem.u0)
    return experiment_filename(experiment.problem.name, d, n, "$(experiment.problem.solver)", id; kwargs...)
end

### experiment result ###
experiment_result_filename(problem_name, d, n, solver, id; dir = "data") = joinpath(dir, "experiment results", lowercase(problem_name), "d_$d", "n_$n", lowercase(solver), "$(id).jld2")
function experiment_result_filename(experiment::GradFlowExperiment, id; kwargs...)
    d, n = size(experiment.problem.u0)
    return experiment_result_filename(experiment.problem.name, d, n, "$(experiment.problem.solver)", id; kwargs...)
end

### metric ###
"Load the avarage of the metric over all the runs for each n and solver."
function load_metric(problem_name, d, ns, solver_names, metric::Symbol)
    metric_matrix = zeros(length(ns), length(solver_names))
    for (i, n) in enumerate(ns), (j, solver_name) in enumerate(solver_names)
        dir = dirname(experiment_result_filename(problem_name, d, n, solver_name, 1))
        filenames = joinpath.(dir, readdir(dir))
        # use the mean of all the runs
        metric_matrix[i, j] = mean([getfield(load(f), metric) for f in filenames])
    end
    return metric_matrix
end

### model ###
model_filename(problem_name, d, n; dir = "data") = joinpath(dir, "models", lowercase(problem_name), "d_$(d)", "n_$(n).jld2")
function best_model(problem_name, d; kwargs...)
    dir = dirname(model_filename(problem_name, d, 1; kwargs...))
    filenames = readdir(dir)
    if isempty(filenames)
        return nothing
    end
    get_n(f) = parse(Int, f[3:findfirst('.', f)-1])
    highest_n_filename = maximum(f -> (get_n(f), f), filenames)[2]
    return load(joinpath(dir, highest_n_filename))
end

### timer ###
timer_filename(problem_name, d, dir = "data") = joinpath(dir, "timers", lowercase(problem_name), "d_$d", "timer.jld2")

### Pretty printing ###
short_string(float::Number, width, digits=width - 2) = rpad(round(float, digits=digits), width)