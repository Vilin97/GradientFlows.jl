experiment_filename(problem_name, solver, d, n, id; path=joinpath("data", "experiments")) = joinpath(path, lowercase(problem_name), lowercase(solver), "d_$(d)", "n_$(n)", "$(id).jld2")
function experiment_filename(experiment::GradFlowExperiment, id; kwargs...)
    d, n = size(experiment.problem.u0)
    return experiment_filename(experiment.problem.name, "$(experiment.problem.solver)", d, n, id; kwargs...)
end

model_filename(problem_name, d, n; path=joinpath("data", "models")) = joinpath(path, lowercase(problem_name), "d_$(d)", "n_$(n).jld2")

save(path, obj) = (mkpath(dirname(path)); JLD2.save_object(path, obj))
load(path) = JLD2.load_object(path)

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

### Pretty printing ###
short_string(float::Number, width, digits=width-2) = rpad(round(float, digits=digits), width)