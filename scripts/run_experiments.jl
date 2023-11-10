using GradientFlows, TimerOutputs

"""
run_experiments(problems, ns, num_runs; verbose=true)

Run experiments for all the problems and save the results.
    problems = [(problem, d), ...]
    ns = [n, ...]
    num_runs = number of runs for each experiment
"""
function run_experiments(problems, ns, num_runs; verbose=true)
    timer = TimerOutput()

    # generate data
    for n in ns
        @timeit timer "n $n" for (problem, d) in problems
            @timeit timer "d $d" for run in 1:num_runs
                prob_ = problem(d, n, Exact())
                problem_name = prob_.name
                solvers = [Blob(blob_epsilon(d, n)), SBTM(best_model(problem_name, d)), Exact()]
                @timeit timer "$problem_name" for solver in solvers
                    prob = problem(d, n, solver)
                    set_u0!(prob, prob_.u0)
                    experiment = GradFlowExperiment(prob)
                    @time @timeit timer "$solver" solve!(experiment)
                    
                    save(experiment_filename(experiment, run), experiment)
                    
                    if verbose 
                        println("$experiment")
                    end
                end
            end
        end
    end

    # analyze data and merge timers
    for n in ns, (problem, d) in problems
        prob_ = problem(d, n, Exact())
        problem_name = prob_.name
        solvers = [Blob(blob_epsilon(d, n)), SBTM(best_model(problem_name, d)), Exact()]
        for run in 1:num_runs, solver in solvers
            experiment = load(experiment_filename(problem_name, d, n, "$solver", run))
            @timeit timer "analyze" result = GradFlowExperimentResult(experiment)
            save(experiment_result_filename(problem_name, d, n, "$solver", run), result)
            merge!(timer["n $n"]["d $d"][problem_name]["$solver"], experiment.timer)
        end
    end

    save(timer_filename(), timer)
    println(timer)
    nothing
end

function train_nn(problem, d, n, s; verbose=1, init_max_iterations=10^5)
    solver = SBTM(s, logger=Logger(verbose), init_max_iterations=init_max_iterations)
    prob = problem(d, n, solver)
    @time train_s!(solver, prob.u0, score(prob.ρ0, prob.u0))
    save(model_filename(prob.name, d, n), solver.s)
    nothing
end

### train NN ###
println("Training NNs")
problems = [(3, landau_problem, "landau"), (5, landau_problem, "landau"), (10, landau_problem, "landau")]
models = [mlp(3;depth=2), mlp(5;depth=2), mlp(10;depth=2)]
for ((d, problem, problem_name), model) in zip(problems, models)
    @show d, problem_name
    train_nn(problem, d, 40_000, model; init_max_iterations=10^6)
end

### generate data ###
println("Generating data")
problems = [(diffusion_problem, 2), (diffusion_problem, 5), (landau_problem, 3), (landau_problem, 5), (landau_problem, 10)]
num_runs = 5
ns = 100 * 2 .^ (0:8)
run_experiments(problems, ns, num_runs)

### plot ###
println("Plotting")
include("plot.jl")
solver_names = ["exact", "sbtm", "blob"]
problems = [(2, "diffusion"), (5, "diffusion"), (3, "landau"), (5, "landau"), (10, "landau")]
for (d, problem_name) in problems
    @show d, problem_name
    @time plot_all(problem_name, d, ns, solver_names)
end