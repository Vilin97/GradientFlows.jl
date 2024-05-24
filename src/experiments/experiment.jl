mutable struct Experiment{D,S,C,P,V,T,M}
    true_dist::Vector{D}
    true_cov::Vector{M}
    solution::Vector{S}
    score_values::Vector{C}
    velocity_values::Vector{C}
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
    velocity_values = (!hasfield(typeof(problem.solver), :logger) || isempty(problem.solver.logger.velocity_values)) ? fill(NaN, length(solution)) : problem.solver.logger.velocity_values
    timer = deepcopy(DEFAULT_TIMER)
    true_dist_ = [true_dist(problem, t) for t in saveat]
    true_cov = [problem.covariance(t, problem.params) for t in saveat]
    return Experiment(true_dist_, true_cov, solution, score_values, velocity_values, problem.params, saveat, Float64(problem.dt), timer, problem.name, name(problem.solver))
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
function run_experiments(problems, ns, runs, solvers; rng=StableRNG, verbose=1, model_dir="data", dir="data", kwargs...)

Run experiments for all the problems and save the results.
    problems = [(problem, d), ...], where problem(d, n, solver; kwargs...)
    ns = [n, ...], number of particles
    runs = run ids, e.g. 1:5 means five runs
    solvers = [Blob(), ...]
"""
function run_experiments(problems, ns, runs, solvers; rng=StableRNG, verbose=1, model_dir="data", dir="data", kwargs...)

    verbose > 0 && (@info "Generating data")
    timer = TimerOutput()

    for n in ns
        @timeit timer "n $n" for (problem, d) in problems
            @timeit timer "d $d" for run in runs
                prob_ = problem(d, n, Exact(); rng=rng(100 * d + run))
                problem_name = prob_.name
                @timeit timer "$problem_name" for solver in solvers
                    prob = problem(d, n, solver; dir=model_dir, kwargs...)
                    set_u0!(prob, prob_.u0)
                    if verbose > 0 && run == runs[end]
                        @info "n=$(rpad(n,5)) $(rpad(problem_name, PROBLEM_NAME_WIDTH)) d=$(rpad(d,2)) run=$run solver=$solver"
                        @time @timeit timer "$solver" experiment = Experiment(prob)
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
        @warn e
    end
    save(timer_filename(; dir=dir), timer)
    nothing
end
