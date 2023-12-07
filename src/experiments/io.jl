### generic ###
save(path, obj) = (mkpath(dirname(path)); JLD2.save_object(path, obj))
load(path) = JLD2.load_object(path)

### experiment ###
experiment_filename(problem_name, d, n, solver, id; dir) = joinpath(dir, "experiments", lowercase(problem_name), "d_$d", "n_$n", lowercase(solver), "$(id).jld2")
function experiment_filename(experiment::Experiment, id; kwargs...)
    d, n = size(experiment.solution[1])
    return experiment_filename(experiment.problem_name, d, n, experiment.solver_name, id; kwargs...)
end

### experiment result ###
experiment_result_filename(problem_name, d, n, solver, id; dir) = joinpath(dir, "experiment results", lowercase(problem_name), "d_$d", "n_$n", lowercase(solver), "$(id).jld2")
function experiment_result_filename(experiment::Experiment, id; kwargs...)
    d, n = size(experiment.solution[1])
    return experiment_result_filename(experiment.problem_name, d, n, experiment.solver_name, id; kwargs...)
end

### metric ###
"Load the average of the metric over all the runs for each n and solver.

    metric_matrix[i, j] = metric(n = ns[i], solver = solver_names[j])"
function load_metric(problem_name, d, ns, solver_names, metric::Symbol; kwargs...)
    metric_matrix = zeros(length(ns), length(solver_names))
    for (i, n) in enumerate(ns), (j, solver_name) in enumerate(solver_names)
        dir = dirname(experiment_result_filename(problem_name, d, n, solver_name, 1; kwargs...))
        filenames = joinpath.(dir, readdir(dir))
        # use the mean of all the runs
        metric_matrix[i, j] = mean([getfield(load(f), metric) for f in filenames])
    end
    return metric_matrix
end

### all runs of experiment ###
function load_all_experiment_runs(problem_name, d, n, solver_name; kwargs...)
    dir = dirname(experiment_filename(problem_name, d, n, solver_name, 1; kwargs...))
    return load.(joinpath.(dir, readdir(dir)))
end

### model ###
model_filename(problem_name, d, n; dir="data") = joinpath(dir, "models", lowercase(problem_name), "d_$(d)", "n_$(n).jld2")
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
timer_filename(;dir) = joinpath(dir, "timers", "timer.jld2")

### Pretty printing ###
pretty(x::Number, width) = rpad(string(x)[1:width], width)