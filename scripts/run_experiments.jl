using GradientFlows, TimerOutputs, StableRNGs

"""
run_experiments(problems, ns, num_runs; verbose=true, dt=0.01, dir="data")

Run experiments for all the problems and save the results.
    problems = [(problem, d), ...]
    ns = [n, ...]
    num_runs = number of runs for each experiment
"""
function run_experiments(problems, ns, num_runs, solvers; dt=0.01, dir="data")

    println("Generating data")
    timer = TimerOutput()

    for n in ns
        @timeit timer "n $n" for (problem, d) in problems
            @timeit timer "d $d" for run in 1:num_runs
                prob_ = problem(d, n, Exact(); rng=StableRNG(100*n + 10*d + run))
                problem_name = prob_.name
                @timeit timer "$problem_name" for solver in solvers
                    prob = problem(d, n, solver; dt=dt)
                    set_u0!(prob, prob_.u0)
                    if run==num_runs
                        @time "n=$n $problem_name d=$(rpad(d,2)) run=$run solver=$solver" @timeit timer "$solver" experiment = Experiment(prob)
                    else
                        @timeit timer "$solver" experiment = Experiment(prob)
                    end
                    save(experiment_filename(experiment, run; dir=dir), experiment)

                    @timeit timer "analyze" result = GradFlowExperimentResult(experiment)
                    save(experiment_result_filename(problem_name, d, n, "$solver", run; dir=dir), result)
                    merge!(timer["n $n"]["d $d"][problem_name]["$solver"], experiment.timer)
                end
            end
        end
    end

    try
        old_timer = GradientFlows.load(timer_filename(; dir=dir))
        merge!(timer, old_timer)
    catch e
        println(e)
    end
    save(timer_filename(; dir=dir), timer)
    nothing
end

function train_nn(problem, d, n, s; verbose=1, init_max_iterations=10^5)
    solver_ = SBTM(s, logger=Logger(verbose), init_max_iterations=init_max_iterations)
    prob = problem(d, n, solver_)
    println("Training NN for $(prob.name), d = $d, n = $n.")
    @time train_s!(prob.solver, prob.u0, score(prob.ρ0, prob.u0))
    save(model_filename(prob.name, d, n), prob.solver.s)
    nothing
end

### generate data ###
problems = [(fpe_problem, 2), (fpe_problem, 5), (fpe_problem, 10)]
num_runs = 5
ns = 100 * 2 .^ (0:8)
solvers = ALL_SOLVERS

# run_experiments(problems, ns, num_runs, solvers)

include("plot.jl")
@time plot_all(problems, ns, ALL_SOLVERS);
