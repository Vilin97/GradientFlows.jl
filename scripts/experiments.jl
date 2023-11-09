using GradientFlows, TimerOutputs

function run_experiment(problem, d, ns, num_runs; verbose=true)
    reset_timer!(DEFAULT_TIMER)
    timer = TimerOutput()
    for n in ns
        for run in 1:num_runs
            prob_ = problem(d, n, Exact())
            solvers = [Blob(blob_epsilon(d, n)), SBTM(best_model(prob_.name, d)), Exact()]
            @show run
            for solver in solvers
                prob = problem(d, n, solver)
                set_u0!(prob, prob_.u0)
                experiment = GradFlowExperiment(prob)
                solve!(experiment)
                result = GradFlowExperimentResult(experiment)
                save(experiment_filename(experiment, run), experiment)
                save(experiment_result_filename(experiment, run), result)
                merge!(timer, experiment.timer)
                verbose && println("    $experiment")
            end
        end
    end
    problem_name = problem(d, ns[1], Exact()).name
    save(timer_filename(problem_name, d), timer)
    println(timer)
    nothing
end

function train_nn(problem, d, n, s; verbose=1, init_max_iterations=10^6)
    solver = SBTM(s, logger=Logger(verbose), init_max_iterations=init_max_iterations)
    prob = problem(d, n, solver)
    @time train_s!(solver, prob.u0, score(prob.ρ0, prob.u0))
    save(model_filename(prob.name, d, n), solver.s)
    nothing
end

## train NN
# problems = [(2, diffusion_problem, "diffusion"), (5, diffusion_problem, "diffusion"), (3, landau_problem, "landau"), (5, landau_problem, "landau"), (10, landau_problem, "landau")]
# for (d, problem, problem_name) in problems
#     @show d, problem_name
#     train_nn(problem, d, 80_000, best_model(problem_name, d))
# end

## run experiments
# problems = [(2, diffusion_problem, "diffusion"), (5, diffusion_problem, "diffusion"), (3, landau_problem, "landau"), (5, landau_problem, "landau"), (10, landau_problem, "landau")]

num_runs = 5
@show 3, "landau"
@time run_experiment(landau_problem, 3, 100 * 2 .^ (6:8), num_runs)

ns = 100 * 2 .^ (0:8)
problems = [(5, landau_problem, "landau"), (10, landau_problem, "landau")]
for (d, problem, problem_name) in problems
    @show d, problem_name
    @time run_experiment(problem, d, ns, num_runs)
end