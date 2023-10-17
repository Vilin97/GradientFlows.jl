experiment_filename(problem_name, solver_type, d, n, num_solutions) = "data/experiments/$(lowercase(problem_name))/$(lowercase(solver_type))/d_$(d)/n_$(n)_runs_$(num_solutions).jld2"

"Save the experiment to a .jld2 file."
function save(experiment::GradFlowExperiment)
    problem = experiment.problem
    solver = problem.solver
    d, n = size(problem.u0)
    num_solutions = length(experiment.solutions)
    filename = experiment_filename(problem.name, typeof(solver), d, n, num_solutions)
    JLD2.save(filename, "experiment", experiment)
end

function load_experiment(problem_name, solver_type, d, n, num_solutions)
    filename = experiment_filename(problem_name, solver_type, d, n, num_solutions)
    return JLD2.load(filename, "experiment")
end

model_filename(problem_name, d, n) = "data/models/$(lowercase(problem_name))/d_$(d)/n_$(n).jld2"

function save(s::Chain, problem_name, d, n)
    filename = model_filename(problem_name, d, n)
    JLD2.save(filename, "nn", s)
end

function load_model(problem_name, d, n)
    filename = model_filename(problem_name, d, n)
    return JLD2.load(filename, "nn")
end