mutable struct Experiment{D,S,C,P,V,T,M}
    true_dist::Vector{D}
    true_cov::Vector{M}
    solution::Vector{S}
    score_values::Vector{C}
    params::P
    saveat::V
    dt::Float64
    timer::T
    problem_name::String
    solver_name::String
end

function Experiment(problem::GradFlowProblem; saveat=collect(problem.tspan[1]:problem.dt:problem.tspan[2]))
    reset_timer!(DEFAULT_TIMER)
    solution = solve(problem, saveat=saveat).u
    score_values = (!hasfield(typeof(problem.solver), :logger) || isempty(problem.solver.logger.score_values)) ? fill(NaN, length(solution)) : problem.solver.logger.score_values
    timer = deepcopy(DEFAULT_TIMER)
    true_dist_ = [true_dist(problem, t) for t in saveat]
    true_cov = [problem.covariance(t, problem.params) for t in saveat]
    return Experiment(true_dist_, true_cov, solution, score_values, problem.params, saveat, Float64(problem.dt), timer, problem.name, name(problem.solver))
end

function Base.show(io::IO, experiment::Experiment)
    @unpack solution, problem_name, solver_name = experiment
    d, n = size(solution[1])
    print(io, "$problem_name d=$d n=$(rpad(n,n_WIDTH)) $(lpad(solver_name, SOLVER_NAME_WIDTH))")
end

have_true_dist(experiment::Experiment) = !isnothing(experiment.true_dist[end])

mean_conserved(experiment::Experiment) = !have_true_dist(experiment) || (mean(experiment.true_dist[end]) ≈ mean(experiment.true_dist[1]))
cov_trace_conserved(experiment::Experiment) = !have_true_dist(experiment) || (tr(cov(experiment.true_dist[end])) ≈ tr(cov(experiment.true_dist[1])))

"""
run_experiments(problems, ns, num_runs; verbose=true, dt=0.01, dir="data")

Run experiments for all the problems and save the results.
    problems = [(problem, d), ...]
    ns = [n, ...]
    num_runs = number of runs for each experiment
"""
function run_experiments(problems, ns, num_runs, solvers; rng=StableRNG, verbose=1, dt=0.01, dir="data")

    verbose > 0 && println("Generating data")
    timer = TimerOutput()

    for n in ns
        @timeit timer "n $n" for (problem, d) in problems
            @timeit timer "d $d" for run in 1:num_runs
                prob_ = problem(d, n, Exact(); rng=rng(100 * n + 10 * d + run))
                problem_name = prob_.name
                @timeit timer "$problem_name" for solver in solvers
                    prob = problem(d, n, solver; dt=dt, dir=dir)
                    set_u0!(prob, prob_.u0)
                    if verbose > 0 && run == num_runs
                        @time "n=$(rpad(n,5)) $problem_name d=$(rpad(d,2)) run=$run solver=$solver" @timeit timer "$solver" experiment = Experiment(prob)
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

function train_nn(problem, d, n, s; verbose=1, init_max_iterations=10^5, dir="data")
    solver_ = SBTM(s, verbose=verbose, init_max_iterations=init_max_iterations)
    prob = problem(d, n, solver_)
    verbose > 0 && println("Training NN for $(prob.name), d = $d, n = $n.")
    @time train_s!(prob.solver, prob.u0, score(prob.ρ0, prob.u0))
    save(model_filename(prob.name, d, n; dir=dir), prob.solver.s)
    nothing
end