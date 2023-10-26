using GradientFlows, TimerOutputs

function run_experiment(problem, d, ns, num_runs; verbose=true)
    reset_timer!(DEFAULT_TIMER)
    timer = TimerOutput()
    for n in ns
        for run in 1:num_runs
            prob_ = problem(d, n, Exact())
            solvers = [Blob(blob_eps(d, n)), SBTM(best_model(prob_.name, d)), Exact()]
            @show run
            for solver in solvers
                prob = problem(d, n, solver)
                set_u0!(prob, prob_.u0)
                experiment = GradFlowExperiment(prob)
                solve!(experiment)
                compute_errors!(experiment)
                save(experiment_filename(experiment, run), experiment)
                merge!(timer, experiment.timer)
                verbose && println("    $experiment")
            end
        end
    end
    save(timer_filename(problem(d, ns[1], Exact()).name, d), timer)
    println(timer)
    nothing
end

function train_nn(problem, d, n, s; verbose=2, init_max_iterations=10^6)
    solver = SBTM(s, logger=Logger(verbose), init_max_iterations=init_max_iterations)
    prob = problem(d, n, solver)
    @time train_s!(solver, prob.u0, score(prob.ρ0, prob.u0))
    save(model_filename(prob.name, d, n), solver.s)
    nothing
end

d = 5
# train_nn(landau_problem, d, 1000, mlp(d, depth=2))
run_experiment(landau_problem, d, [100], 1)