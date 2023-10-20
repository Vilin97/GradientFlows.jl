using GradientFlows
using TimerOutputs

reset_timer!(DEFAULT_TIMER)
d = 5
ns = [10000, 20000, 40000]
num_solutions = 10
verbose = true
save_intermediates = true
for (j,n) in enumerate(ns)
    model = best_model("landau", d)
    solvers = [Blob(blob_eps(d, n)), SBTM(model), Exact()]
    for (i,solver) in enumerate(solvers)
        @timeit DEFAULT_TIMER "d=$d n=$(rpad(n, 6)) setup $solver" problem = landau_problem(d, n, solver)
        experiment = GradFlowExperiment(problem, num_solutions)
        solve!(experiment)
        compute_errors!(experiment)

        if verbose
            print("$experiment")
        end
        if save_intermediates
            save(experiment_filename(experiment), experiment)
            # TODO make io for timer
            problem_name = experiment.problem.name
            save("data/experiments/$problem_name/timer.jld2", DEFAULT_TIMER)
        end
    end
end
println(DEFAULT_TIMER)